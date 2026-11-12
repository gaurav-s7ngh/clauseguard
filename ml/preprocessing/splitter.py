"""
ClauseGuard Contract-Level Dataset Splitter
===========================================
Partitions contracts into strictly disjoint train (340), validation (68), and test (102) sets.
Ensures zero data leakage across document boundaries.
"""

from typing import Dict, Set, List, Tuple
import numpy as np
from ml.preprocessing.constants import (
    RANDOM_SEED,
    EXPECTED_TOTAL_CONTRACTS,
    EXPECTED_OFFICIAL_TRAIN_CONTRACTS,
    EXPECTED_OFFICIAL_TEST_CONTRACTS,
    TARGET_TRAIN_CONTRACTS,
    TARGET_VAL_CONTRACTS,
)

def create_contract_splits(
    official_train_titles: List[str],
    official_test_titles: List[str],
    random_seed: int = RANDOM_SEED
) -> Dict[str, str]:
    """
    Creates a deterministic contract-level split:
    - 340 contracts for training
    - 68 contracts for validation
    - 102 contracts for testing (official test set, preserved untouched)

    Returns:
        Dict mapping contract_title -> split_name ("train", "val", "test")
    """
    train_pool = sorted(list(set(official_train_titles)))
    test_pool = sorted(list(set(official_test_titles)))

    # Verification of input contracts
    if len(train_pool) != EXPECTED_OFFICIAL_TRAIN_CONTRACTS:
        raise ValueError(
            f"Expected {EXPECTED_OFFICIAL_TRAIN_CONTRACTS} official train contracts, got {len(train_pool)}"
        )
    if len(test_pool) != EXPECTED_OFFICIAL_TEST_CONTRACTS:
        raise ValueError(
            f"Expected {EXPECTED_OFFICIAL_TEST_CONTRACTS} official test contracts, got {len(test_pool)}"
        )

    # Verification of zero leakage between official train and test
    overlap = set(train_pool).intersection(set(test_pool))
    if overlap:
        raise ValueError(f"Data leakage detected between train and test pools: {len(overlap)} overlapping contracts!")

    # Deterministic seeded partition of the 408 training contracts
    rng = np.random.RandomState(random_seed)
    shuffled_indices = np.arange(len(train_pool))
    rng.shuffle(shuffled_indices)

    val_indices = shuffled_indices[:TARGET_VAL_CONTRACTS]
    train_indices = shuffled_indices[TARGET_VAL_CONTRACTS:]

    val_set: Set[str] = {train_pool[i] for i in val_indices}
    train_set: Set[str] = {train_pool[i] for i in train_indices}
    test_set: Set[str] = set(test_pool)

    # Validate split sizes
    assert len(train_set) == TARGET_TRAIN_CONTRACTS, f"Train count mismatch: {len(train_set)}"
    assert len(val_set) == TARGET_VAL_CONTRACTS, f"Val count mismatch: {len(val_set)}"
    assert len(test_set) == EXPECTED_OFFICIAL_TEST_CONTRACTS, f"Test count mismatch: {len(test_set)}"

    # Validate disjointness
    assert len(train_set.intersection(val_set)) == 0, "Train and Val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == EXPECTED_TOTAL_CONTRACTS

    split_map: Dict[str, str] = {}
    for t in train_set:
        split_map[t] = "train"
    for t in val_set:
        split_map[t] = "val"
    for t in test_set:
        split_map[t] = "test"

    return split_map
