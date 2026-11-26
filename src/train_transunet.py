import argparse
import ast
import copy
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
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
        "--dataset", required=True,
        choices=["QaTa-COV19", "MosMedDataPlus"],
    )
    parser.add_argument(
        "--seed", required=True, type=int,
        choices=[1001, 1002, 1003],
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    os.chdir(project)

    config_text = (project / "configs/transunet.yaml").read_text()
    config = yaml.safe_load(config_text)
    training = config["training"]

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside a GPU allocation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)

    run_dir = (
        project / "runs/transunet"
        / args.dataset / f"seed_{args.seed}"
    )
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
                "Use --resume only for an unfinished run."
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
    if (len(train_set), len(val_set)) != expected[args.dataset]:
        raise RuntimeError("Unexpected training/validation dataset sizes.")

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(
        batch_size=training["batch_size"],
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=False, **loader_options
    )
    val_loader = DataLoader(
        val_set, shuffle=False, drop_last=False, **loader_options
    )

    sys.path.insert(0, str(project / "external/TransUNet"))
    from networks.vit_seg_modeling import VisionTransformer, CONFIGS

    model_config = copy.deepcopy(
        CONFIGS[config["model"]["configuration"]]
    )
    model_config.n_classes = config["model"]["num_classes"]
    model_config.n_skip = config["model"]["n_skip"]
    model_config.patches.grid = tuple(
        config["model"]["patches_grid"]
    )

    model = VisionTransformer(
        model_config,
        img_size=config["data"]["image_size"][0],
        num_classes=config["model"]["num_classes"],
    )

    # A resumed run gets its weights from last.pt instead.
    if not args.resume:
        with np.load(
            config["model"]["pretrained_path"], allow_pickle=False
        ) as weights:
            model.load_from(weights)

    model = model.cuda()

    # Load only the official DiceLoss class.
    utils_path = project / "external/TransUNet/utils.py"
    tree = ast.parse(utils_path.read_text())
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DiceLoss"
    ]
    if len(classes) != 1:
        raise RuntimeError("Could not locate the official DiceLoss.")

    namespace = {"torch": torch, "nn": nn, "np": np}
    exec(
        compile(
            ast.Module(body=classes, type_ignores=[]),
            str(utils_path), "exec",
        ),
        namespace,
    )
    dice_loss = namespace["DiceLoss"](config["model"]["num_classes"])
    ce_loss = nn.CrossEntropyLoss()
    loss_config = training["loss"]

    def criterion(logits, labels):
        return (
            loss_config["cross_entropy_weight"]
            * ce_loss(logits, labels)
            + loss_config["dice_weight"]
            * dice_loss(
                logits, labels,
                softmax=loss_config["dice_softmax"],
            )
        )

    opt = training["optimizer"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt["learning_rate"],
        momentum=opt["momentum"],
        weight_decay=opt["weight_decay"],
    )

    max_epochs = training["max_epochs"]
    total_iterations = max_epochs * len(train_loader)
    power = training["scheduler"]["power"]
    start_epoch = 1
    global_step = 0
    best_dice = -1.0
    best_epoch = 0

    if args.resume:
        # Load only your own trusted training checkpoint.
        state = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        if (
            state["dataset"] != args.dataset
            or state["seed"] != args.seed
            or state["total_iterations"] != total_iterations
            or state["workers"] != args.workers
        ):
            raise RuntimeError("Resume settings differ from the saved run.")

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_dice = state["best_dice"]
        best_epoch = state["best_epoch"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
        del state

    fields = [
        "epoch", "lr_first", "lr_last",
        "train_loss", "val_loss", "val_dice", "seconds",
    ]

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

    print(
        f"Dataset={args.dataset}, seed={args.seed}, "
        f"train={len(train_set)}, val={len(val_set)}, "
        f"batch_size={training['batch_size']}, "
        f"epochs={max_epochs}, "
        f"GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    with history_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
            handle.flush()

        for epoch in range(start_epoch, max_epochs + 1):
            started = time.perf_counter()
            model.train()
            train_loss = 0.0
            lr_first = None

            for batch in train_loader:
                # Set the learning rate used by this optimizer update.
                lr = opt["learning_rate"] * (
                    1.0 - global_step / total_iterations
                ) ** power
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if lr_first is None:
                    lr_first = lr

                images = batch["image"].cuda(non_blocking=True)
                labels = batch["mask"][:, 0].long().cuda(
                    non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Nonfinite training loss at epoch {epoch}"
                    )

                loss.backward()
                optimizer.step()
                global_step += 1
                train_loss += loss.item() * images.size(0)

            lr_last = lr
            model.eval()
            val_loss = 0.0
            dice_sum = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].cuda(non_blocking=True)
                    labels = batch["mask"][:, 0].long().cuda(
                        non_blocking=True
                    )
                    logits = model(images)
                    loss = criterion(logits, labels)

                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite validation loss.")

                    val_loss += loss.item() * images.size(0)

                    foreground = torch.softmax(logits, dim=1)[:, 1]
                    pred = foreground > config["validation"][
                        "foreground_threshold"
                    ]
                    target = labels == 1

                    intersection = (pred & target).sum((1, 2)).float()
                    denominator = (
                        pred.sum((1, 2)) + target.sum((1, 2))
                    )
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
                best_dice = val_dice
                best_epoch = epoch

            state = {
                "epoch": epoch,
                "global_step": global_step,
                "total_iterations": total_iterations,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_dice": best_dice,
                "best_epoch": best_epoch,
                "dataset": args.dataset,
                "seed": args.seed,
                "workers": args.workers,
                "config": config,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "loader_rng": generator.get_state(),
            }

            # Save the resumable state before publishing the best weights.
            save_checkpoint(last_path, state)
            if improved:
                save_checkpoint(run_dir / "best.pt", {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_dice": best_dice,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "config": config,
                })

            seconds = time.perf_counter() - started
            writer.writerow({
                "epoch": epoch,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "seconds": seconds,
            })
            handle.flush()

            print(
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"lr={lr_last:.7f} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_dice={val_dice:.6f} | "
                f"best={best_dice:.6f} | "
                f"seconds={seconds:.1f}",
                flush=True,
            )

    # Repair best.pt if a previous interruption happened between saves.
    if last_path.exists():
        best_path = run_dir / "best.pt"
        saved_best_epoch = -1
        if best_path.exists():
            saved_best = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            saved_best_epoch = saved_best["epoch"]
            del saved_best
        if saved_best_epoch != best_epoch:
            state = torch.load(
                last_path, map_location="cpu", weights_only=False
            )
            if state["epoch"] != best_epoch:
                raise RuntimeError(
                    "best.pt does not match the recorded best epoch; "
                    "inspect checkpoints before evaluation."
                )
            save_checkpoint(best_path, {
                key: state[key] for key in [
                    "epoch", "model", "best_dice",
                    "dataset", "seed", "config",
                ]
            })

    print(
        f"Training finished. Best validation Dice: {best_dice:.6f}, "
        f"best epoch: {best_epoch}",
        flush=True,
    )


if __name__ == "__main__":
    main()