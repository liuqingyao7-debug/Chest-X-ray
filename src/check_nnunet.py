import gc
import json
import os
import random
from pathlib import Path

# Lightweight smoke-test settings; do not change the main plans.
os.environ["nnUNet_compile"] = "false"
os.environ["nnUNet_n_proc_DA"] = "0"

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

if not torch.cuda.is_available():
    raise RuntimeError("Run inside a GPU allocation.")

torch.set_num_threads(2)
root = Path(os.environ["nnUNet_preprocessed"])
summary = {}

for name in ["Dataset501_QaTa", "Dataset502_MosMed"]:
    random.seed(1001)
    np.random.seed(1001)
    torch.manual_seed(1001)
    torch.cuda.manual_seed_all(1001)

    folder = root / name
    plans = json.loads((folder / "nnUNetPlans.json").read_text())
    dataset_json = json.loads((folder / "dataset.json").read_text())
    split = json.loads((folder / "splits_final.json").read_text())[0]

    print(f"\n=== {name} ===", flush=True)
    plans["continue_training"] = False
    trainer = nnUNetTrainer(
        plans=plans,
        configuration="2d",
        fold=0,
        dataset_json=dataset_json,
        device=torch.device("cuda", 0),
    )

    # Keep smoke-test outputs separate from main experiments.
    output = Path("runs/setup_checks/nnunet") / name
    output.mkdir(parents=True, exist_ok=True)
    trainer.output_folder_base = str(output)
    trainer.output_folder = str(output / "fold_0")
    Path(trainer.output_folder).mkdir(parents=True, exist_ok=True)

    trainer.initialize()
    trainer.batch_size = 2

    # Use only a tiny subset, preserving train/validation membership.
    trainer.do_split = lambda: (split["train"][:4], split["val"][:2])
    train_loader, val_loader = trainer.get_dataloaders()
    trainer.network.train()

    batch = next(train_loader)
    targets = batch["target"]
    target = targets[0] if isinstance(targets, list) else targets
    print("Input shape:", tuple(batch["data"].shape), flush=True)
    print("Full-resolution target:", tuple(target.shape), flush=True)

    assert tuple(batch["data"].shape) == (2, 3, 224, 224)
    assert tuple(target.shape) == (2, 1, 224, 224)

    with torch.no_grad(), torch.autocast("cuda"):
        outputs = trainer.network(batch["data"].cuda())
    outputs_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    assert tuple(outputs_list[0].shape) == (2, 2, 224, 224)
    assert all(torch.isfinite(item).all() for item in outputs_list)
    print("Output shapes:", [tuple(item.shape) for item in outputs_list],
          flush=True)
    del outputs, outputs_list

    losses = []
    for step in range(10):
        result = trainer.train_step(next(train_loader))
        loss = float(result["loss"])
        assert np.isfinite(loss), "Nonfinite loss"

        grads = [
            p.grad for p in trainer.network.parameters()
            if p.grad is not None
        ]
        assert grads, "No gradients"
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g).item() > 0 for g in grads)

        losses.append(loss)
        print(f"Step {step + 1:02d}/10: loss={loss:.6f}", flush=True)

    summary[name] = {
        "status": "passed",
        "smoke_batch_size": 2,
        "planned_batch_size": plans["configurations"]["2d"]["batch_size"],
        "losses": losses,
        "initial_lr": trainer.initial_lr,
        "weight_decay": trainer.weight_decay,
        "max_epochs": trainer.num_epochs,
        "optimizer": type(trainer.optimizer).__name__,
        "scheduler": type(trainer.lr_scheduler).__name__,
    }
    print("PASS: native data loading, output shapes, loss, gradients, "
          "10 training steps", flush=True)

    del trainer, train_loader, val_loader, batch, targets, target, grads
    gc.collect()
    torch.cuda.empty_cache()

output = Path("reports/setup_checks/nnunet_smoke_summary.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2) + "\n")
print(f"\nSummary saved: {output}", flush=True)
