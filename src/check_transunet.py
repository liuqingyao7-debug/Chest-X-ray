import ast
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from phase0_dataset import Phase0Dataset

random.seed(1001)
np.random.seed(1001)
torch.manual_seed(1001)
torch.set_num_threads(2)

if not torch.cuda.is_available():
    raise RuntimeError("Run this script inside a GPU allocation.")

torch.cuda.manual_seed_all(1001)
sys.path.insert(0, str(Path("external/TransUNet").resolve()))
from networks.vit_seg_modeling import VisionTransformer, CONFIGS

# Load the exact official DiceLoss class without unrelated dependencies.
path = Path("external/TransUNet/utils.py")
tree = ast.parse(path.read_text())
classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "DiceLoss"
]
assert len(classes) == 1
namespace = {"torch": torch, "nn": nn, "np": np}
exec(compile(ast.Module(body=classes, type_ignores=[]),
             str(path), "exec"), namespace)
dice_loss = namespace["DiceLoss"](2)
ce_loss = nn.CrossEntropyLoss()

for name in ["QaTa-COV19", "MosMedDataPlus"]:
    print(f"\n=== {name} ===", flush=True)

    config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
    config.n_classes = 2
    config.n_skip = 3
    config.patches.grid = (14, 14)
    model = VisionTransformer(config, img_size=224, num_classes=2)

    with np.load("pretrained/R50+ViT-B_16.npz",
                 allow_pickle=False) as weights:
        model.load_from(weights)

    model = model.cuda()
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01,
        momentum=0.9, weight_decay=0.0001
    )

    dataset = Phase0Dataset(
        "datasets/processed", name, "train", augment=False
    )
    loader = DataLoader(
        Subset(dataset, range(4)),
        batch_size=2, shuffle=False, num_workers=0
    )
    batches = [
        (batch["image"].cuda(), batch["mask"][:, 0].long().cuda())
        for batch in loader
    ]

    for step in range(10):
        images, labels = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        assert logits.shape == (2, 2, 224, 224)
        assert labels.shape == (2, 224, 224)
        assert torch.isfinite(logits).all()

        loss = (
            0.5 * ce_loss(logits, labels)
            + 0.5 * dice_loss(logits, labels, softmax=True)
        )
        assert torch.isfinite(loss), "Nonfinite loss"
        loss.backward()

        grads = [
            p.grad for p in model.parameters() if p.grad is not None
        ]
        assert grads, "No gradients"
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g).item() > 0 for g in grads)
        optimizer.step()

        print(
            f"Step {step + 1:02d}/10: loss={loss.item():.6f}",
            flush=True
        )

    print("PASS: pretrained loading, shapes, loss, gradients, 10 updates",
          flush=True)
    del model, optimizer, logits, loss, grads, batches
    torch.cuda.empty_cache()

print("\nTransUNet smoke checks complete.", flush=True)
