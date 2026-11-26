import ast
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from phase0_dataset import Phase0Dataset

random.seed(1001)
np.random.seed(1001)
torch.manual_seed(1001)
torch.cuda.manual_seed_all(1001)

if not torch.cuda.is_available():
    raise RuntimeError("Run this check inside a GPU allocation.")

# Load the official U-Net without importing unrelated model code.
path = Path("external/LViT/nets/UNet.py")
spec = importlib.util.spec_from_file_location("lvit_unet", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Load the exact three loss classes from the official source.
# This avoids unrelated pandas/sklearn dependencies in utils.py.
path = Path("external/LViT/utils.py")
tree = ast.parse(path.read_text())
names = {"WeightedBCE", "WeightedDiceLoss", "WeightedDiceBCE"}
classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name in names
]
assert {node.name for node in classes} == names
namespace = {"torch": torch, "nn": nn, "F": F}
exec(compile(ast.Module(body=classes, type_ignores=[]),
             str(path), "exec"), namespace)
Loss = namespace["WeightedDiceBCE"]

torch.set_num_threads(2)

for dataset_name in ["QaTa-COV19", "MosMedDataPlus"]:
    print(f"\n=== {dataset_name} ===", flush=True)

    dataset = Phase0Dataset(
        "datasets/processed", dataset_name, "train", augment=False
    )
    loader = DataLoader(
        Subset(dataset, range(4)),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    # Cache a tiny subset on the GPU.
    batches = [
        (batch["image"].cuda(), batch["mask"].cuda())
        for batch in loader
    ]

    model = module.UNet(n_channels=3, n_classes=1).cuda()
    criterion = Loss(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()

    # Compare initial/final loss in the same evaluation mode.
    def subset_loss():
        model.eval()
        with torch.no_grad():
            value = sum(
                criterion(model(x), y).item() for x, y in batches
            ) / len(batches)
        model.train()
        return value

    initial_loss = subset_loss()

    for step in range(10):
        images, masks = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)

        assert predictions.shape == masks.shape == (2, 1, 224, 224)
        assert torch.isfinite(predictions).all()
        assert ((predictions >= 0) & (predictions <= 1)).all()

        loss = criterion(predictions, masks)
        assert torch.isfinite(loss), "Nonfinite loss"
        loss.backward()

        gradients = [
            p.grad for p in model.parameters() if p.requires_grad
        ]
        assert all(g is not None for g in gradients), "Missing gradient"
        assert all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g).item() > 0 for g in gradients)

        optimizer.step()
        print(
            f"Step {step + 1:02d}/10: loss={loss.item():.6f}",
            flush=True,
        )

    final_loss = subset_loss()
    assert np.isfinite(initial_loss) and np.isfinite(final_loss)
    print(f"Subset eval loss: {initial_loss:.6f} -> {final_loss:.6f}")
    print("PASS: shapes, probabilities, loss, gradients, 10 updates",
          flush=True)

    del model, optimizer, predictions, loss, batches, gradients
    torch.cuda.empty_cache()

print("\nU-Net smoke checks complete.", flush=True)
