import argparse
import ast
import csv
import importlib.util
import math
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from phase0_dataset import Phase0Dataset


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["QaTa-COV19", "MosMedDataPlus"],
    )
    parser.add_argument("--seed", type=int, choices=[1001, 1002, 1003],
                        required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Resolve all project-relative paths consistently.
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    config_path = project / "configs/unet.yaml"
    config_text = config_path.read_text()
    config = yaml.safe_load(config_text)
    training = config["training"]

    if not torch.cuda.is_available():
        raise RuntimeError("Submit this script to a GPU compute node.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)

    run_dir = project / "runs/unet" / args.dataset / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.csv"

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(last_path)
        if (run_dir / "config.yaml").read_text() != config_text:
            raise RuntimeError("Config changed; restore it before resuming.")
    else:
        if any(run_dir.iterdir()):
            raise RuntimeError(
                f"Run folder is not empty: {run_dir}. "
                "Use --resume to continue an existing run."
            )
        (run_dir / "config.yaml").write_text(config_text)

    train_set = Phase0Dataset(
        config["data"]["root"], args.dataset, "train", augment=True
    )
    val_set = Phase0Dataset(
        config["data"]["root"], args.dataset, "val", augment=False
    )

    expected = {
        "QaTa-COV19": (5716, 1429),
        "MosMedDataPlus": (2183, 273),
    }
    assert (len(train_set), len(val_set)) == expected[args.dataset]

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(
        batch_size=training["batch_size"],
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        val_set, shuffle=False, **loader_options
    )

    # Import the same official model used in check_unet.py.
    model_path = project / config["source"]["model_file"]
    spec = importlib.util.spec_from_file_location("lvit_unet", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.UNet(n_channels=3, n_classes=1).cuda()

    # Load only the required official loss and scheduler classes.
    utils_path = project / "external/LViT/utils.py"
    names = {
        "WeightedBCE", "WeightedDiceLoss", "WeightedDiceBCE",
        "CosineAnnealingWarmRestarts",
    }
    tree = ast.parse(utils_path.read_text())
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    assert {node.name for node in classes} == names
    namespace = {
        "__name__": "lvit_training_components",
        "torch": torch, "nn": nn, "F": F,
        "math": math, "warnings": warnings,
        "_LRScheduler": _LRScheduler,
    }
    exec(
        compile(ast.Module(body=classes, type_ignores=[]),
                str(utils_path), "exec"),
        namespace,
    )

    loss_config = training["loss"]
    criterion = namespace["WeightedDiceBCE"](
        dice_weight=loss_config["dice_weight"],
        BCE_weight=loss_config["BCE_weight"],
    )
    opt = training["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        betas=tuple(opt["betas"]),
        eps=opt["eps"],
        weight_decay=opt["weight_decay"],
    )
    sch = training["scheduler"]
    scheduler = namespace["CosineAnnealingWarmRestarts"](
        optimizer, T_0=sch["T_0"],
        T_mult=sch["T_mult"], eta_min=sch["eta_min"],
    )

    start_epoch, best_dice, bad_epochs = 1, -1.0, 0
    if args.resume:
        # Load only checkpoints created by this training script.
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_dice = state["best_dice"]
        bad_epochs = state["bad_epochs"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])

    print(
        f"Dataset={args.dataset}, seed={args.seed}, "
        f"train={len(train_set)}, val={len(val_set)}, "
        f"GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    fields = ["epoch", "lr", "train_loss", "val_loss", "val_dice"]

    # If interrupted after writing history but before saving last.pt,
    # retain only the epochs represented by the resumable checkpoint.
    if args.resume:
        rows = []
        if history_path.exists():
            with history_path.open(newline="") as handle:
                rows = [
                    row for row in csv.DictReader(handle)
                    if int(row["epoch"]) < start_epoch
                ]
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    with history_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
            handle.flush()

        for epoch in range(start_epoch, training["max_epochs"] + 1):
            if bad_epochs >= training["early_stopping_patience"]:
                break

            lr = optimizer.param_groups[0]["lr"]
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                images = batch["image"].cuda(non_blocking=True)
                masks = batch["mask"].cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                probabilities = model(images)
                loss = criterion(probabilities, masks)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            model.eval()
            val_loss, dice_sum = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].cuda(non_blocking=True)
                    masks = batch["mask"].cuda(non_blocking=True)
                    probabilities = model(images)
                    loss = criterion(probabilities, masks)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite validation loss")
                    val_loss += loss.item() * images.size(0)

                    pred = probabilities > config["validation"]["threshold"]
                    target = masks > 0.5
                    axes = (1, 2, 3)
                    intersection = (pred & target).sum(axes).float()
                    denominator = pred.sum(axes) + target.sum(axes)
                    dice = torch.where(
                        denominator == 0,
                        torch.full_like(
                            intersection,
                            config["validation"][
                                "empty_prediction_and_target_dice"
                            ],
                        ),
                        2.0 * intersection / denominator.clamp_min(1),
                    )
                    dice_sum += dice.sum().item()

            train_loss /= len(train_set)
            val_loss /= len(val_set)
            val_dice = dice_sum / len(val_set)
            improved = val_dice > best_dice
            if improved:
                best_dice, bad_epochs = val_dice, 0
            else:
                bad_epochs += 1

            scheduler.step()

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
                "bad_epochs": bad_epochs,
                "dataset": args.dataset,
                "seed": args.seed,
                "config": config,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "loader_rng": generator.get_state(),
            }
            if improved:
                save_checkpoint(run_dir / "best.pt", state)

            writer.writerow(dict(
                epoch=epoch, lr=lr, train_loss=train_loss,
                val_loss=val_loss, val_dice=val_dice,
            ))
            handle.flush()
            save_checkpoint(last_path, state)

            print(
                f"Epoch {epoch:04d} | lr={lr:.7f} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_dice={val_dice:.6f} | "
                f"best={best_dice:.6f} | "
                f"patience={bad_epochs}/"
                f"{training['early_stopping_patience']}",
                flush=True,
            )

    print(f"Training finished. Best validation Dice: {best_dice:.6f}",
          flush=True)


if __name__ == "__main__":
    main()