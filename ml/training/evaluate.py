"""
ClauseGuard Evaluation & Threshold Calibration Engine
======================================================
Comprehensive evaluation module for multi-label clause classification:
- Computes Macro/Micro Precision, Recall, F1
- Computes per-category Precision, Recall, F1, PR-AUC, Support
- Computes confusion matrices per category
- Calibrates optimal category-specific decision thresholds on validation data
- Freezes thresholds for evaluation on official held-out test data
- Enforces supervision masks (excludes unknown/unverified labels from metric computation)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS

def compute_multilabel_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    thresholds: Optional[Union[float, Dict[str, float], List[float]]] = None,
    label_names: List[str] = LABEL_NAMES
) -> Dict[str, Any]:
    """
    Computes rigorous multi-label evaluation metrics respecting supervision masks.

    Args:
        probs: Array of shape [N, num_labels] with sigmoid probabilities in [0, 1]
        labels: Array of shape [N, num_labels] with binary ground truth in {0, 1}
        label_mask: Array of shape [N, num_labels] with supervision mask in {0, 1}
        thresholds: Float or dict of category thresholds. Default is 0.50.
        label_names: List of category names.

    Returns:
        Dict containing per-category metrics, macro/micro metrics, and support counts.
    """
    N, K = probs.shape

    # Normalize thresholds into array of shape [K]
    if thresholds is None:
        thresh_arr = np.full(K, 0.50, dtype=float)
    elif isinstance(thresholds, (int, float)):
        thresh_arr = np.full(K, float(thresholds), dtype=float)
    elif isinstance(thresholds, dict):
        thresh_arr = np.array([thresholds.get(lbl, 0.50) for lbl in label_names], dtype=float)
    elif isinstance(thresholds, (list, np.ndarray)):
        thresh_arr = np.array(thresholds, dtype=float)
    else:
        raise TypeError(f"Unsupported thresholds type: {type(thresholds)}")

    # Apply thresholds to get binary predictions
    preds = (probs >= thresh_arr).astype(int)

    per_category = {}
    macro_precs, macro_recs, macro_f1s, macro_praucs = [], [], [], []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for k in range(K):
        cat_name = label_names[k]
        mask_k = label_mask[:, k].astype(bool)

        # Evaluate strictly on supervised items
        y_true_k = labels[mask_k, k].astype(int)
        y_pred_k = preds[mask_k, k].astype(int)
        y_prob_k = probs[mask_k, k].astype(float)

        support_pos = int(np.sum(y_true_k))
        support_neg = int(len(y_true_k) - support_pos)
        masked_count = int(N - len(y_true_k))

        tp = int(np.sum((y_pred_k == 1) & (y_true_k == 1)))
        fp = int(np.sum((y_pred_k == 1) & (y_true_k == 0)))
        fn = int(np.sum((y_pred_k == 0) & (y_true_k == 1)))
        tn = int(np.sum((y_pred_k == 0) & (y_true_k == 0)))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        # Precision, recall, F1 with zero division safeguards
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        # PR-AUC / Average Precision
        try:
            if support_pos > 0 and len(y_prob_k) > 0:
                prauc = float(average_precision_score(y_true_k, y_prob_k))
            else:
                prauc = 0.0
        except Exception:
            prauc = 0.0

        macro_precs.append(prec)
        macro_recs.append(rec)
        macro_f1s.append(f1)
        macro_praucs.append(prauc)

        per_category[cat_name] = {
            "threshold": float(thresh_arr[k]),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "pr_auc": round(prauc, 4),
            "support_positive": support_pos,
            "support_negative": support_neg,
            "masked_unknown": masked_count,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }
        }

    # Macro averages
    macro_precision = float(np.mean(macro_precs))
    macro_recall = float(np.mean(macro_recs))
    macro_f1 = float(np.mean(macro_f1s))
    mean_prauc = float(np.mean(macro_praucs))

    # Micro averages
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    return {
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_pr_auc": round(mean_prauc, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "per_category": per_category
    }

def calibrate_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    label_names: List[str] = LABEL_NAMES,
    min_thresh: float = 0.10,
    max_thresh: float = 0.90,
    step: float = 0.02
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Optimizes per-category decision thresholds to maximize validation F1 score.

    Returns:
        optimal_thresholds: Dict mapping category -> float threshold
        best_f1s: Dict mapping category -> float best F1 achieved
    """
    K = probs.shape[1]
    candidate_thresholds = np.arange(min_thresh, max_thresh + 1e-5, step)

    optimal_thresholds = {}
    best_f1s = {}

    for k in range(K):
        cat_name = label_names[k]
        mask_k = label_mask[:, k].astype(bool)

        y_true_k = labels[mask_k, k].astype(int)
        y_prob_k = probs[mask_k, k].astype(float)

        best_f1 = -1.0
        best_t = 0.50

        # If no positive instances, keep 0.50 default
        if np.sum(y_true_k) == 0:
            optimal_thresholds[cat_name] = 0.50
            best_f1s[cat_name] = 0.0
            continue

        for t in candidate_thresholds:
            y_pred = (y_prob_k >= t).astype(int)
            tp = np.sum((y_pred == 1) & (y_true_k == 1))
            fp = np.sum((y_pred == 1) & (y_true_k == 0))
            fn = np.sum((y_pred == 0) & (y_true_k == 1))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        # Fallback safeguard
        if best_f1 <= 0.0:
            best_t = 0.50
            best_f1 = 0.0

        optimal_thresholds[cat_name] = round(best_t, 4)
        best_f1s[cat_name] = round(best_f1, 4)

    return optimal_thresholds, best_f1s

def evaluate_model_on_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    thresholds: Optional[Union[float, Dict[str, float]]] = None,
    pos_weight: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs full inference on a PyTorch DataLoader without computing gradients.

    Returns:
        metrics: Dictionary of computed evaluation metrics
        all_probs: Array [N, num_labels]
        all_labels: Array [N, num_labels]
        all_masks: Array [N, num_labels]
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_mask_elements = 0

    probs_list = []
    labels_list = []
    masks_list = []
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Compute masked loss
            batch_loss = model.compute_loss(logits, labels, label_mask, pos_weight=pos_weight)
            batch_mask_sum = label_mask.sum().item()

            total_loss += batch_loss.item() * max(batch_mask_sum, 1.0)
            total_mask_elements += batch_mask_sum

            # Collect outputs
            probs = model.predict_proba(logits).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(labels.cpu().numpy())
            masks_list.append(label_mask.cpu().numpy())

            if verbose and (batch_idx % 20 == 0 or batch_idx == total_batches):
                print(f"    [Eval] Batch {batch_idx}/{total_batches} ({batch_idx/total_batches*100:.0f}%) complete")

    all_probs = np.vstack(probs_list)
    all_labels = np.vstack(labels_list)
    all_masks = np.vstack(masks_list)

    avg_loss = total_loss / max(total_mask_elements, 1.0)

    # Compute metrics
    metrics = compute_multilabel_metrics(
        probs=all_probs,
        labels=all_labels,
        label_mask=all_masks,
        thresholds=thresholds
    )
    metrics["loss"] = round(float(avg_loss), 4)

    return metrics, all_probs, all_labels, all_masks
