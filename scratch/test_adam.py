import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ml.training.model import ClauseGuardClassifier

model = ClauseGuardClassifier("roberta-base", num_labels=9, attn_implementation="eager")
print("Model loaded")

# Test standard AdamW with foreach=False
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, foreach=False)

x = torch.randint(0, 1000, (4, 256))
mask = torch.ones((4, 256))
out = model(x, mask)
loss = out.sum()
loss.backward()
print("Backward pass done")

try:
    optimizer.step()
    print("AdamW with foreach=False SUCCEEDED!")
except Exception as e:
    import traceback
    print("AdamW with foreach=False failed:")
    traceback.print_exc()
