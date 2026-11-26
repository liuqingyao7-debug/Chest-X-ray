import argparse
import json
import os
import random
import sys
from pathlib import Path

# Set these before importing nnU-Net.
os.environ["nnUNet_compile"] = "false"
os.environ.setdefault("nnUNet_n_proc_DA", "2")

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1")

    project = Path(__file__).resolve().parents[1]
    os.chdir(project)

    # Also make direct invocation resolve the correct dataset paths.
    os.environ["nnUNet_raw"] = str(project / "datasets/nnunet/raw")
    os.environ["nnUNet_preprocessed"] = str(
        project / "datasets/nnunet/preprocessed"
    )
    os.environ["nnUNet_results"] = str(project / "runs/nnunet")

    sys.path.insert(0, str(project / "src/nnunet_custom"))
    from nnUNetTrainerPhase0FullVal import nnUNetTrainerPhase0FullVal

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside a GPU allocation.")

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed)

    config_text = (project / "configs/nnunet.yaml").read_text()
    config = yaml.safe_load(config_text)
    dataset_name = config["datasets"][args.dataset]

    folder = Path(os.environ["nnUNet_preprocessed"]) / dataset_name
    plans_text = (folder / "nnUNetPlans.json").read_text()
    split_text = (folder / "splits_final.json").read_text()
    plans = json.loads(plans_text)
    dataset_json = json.loads((folder / "dataset.json").read_text())
    splits = json.loads(split_text)

    expected = {
        "QaTa-COV19": (5716, 1429),
        "MosMedDataPlus": (2183, 273),
    }
    if len(splits) != 1:
        raise RuntimeError("Expected one fixed train/validation split.")

    split = splits[0]
    if (len(split["train"]), len(split["val"])) != expected[args.dataset]:
        raise RuntimeError("Unexpected split sizes.")
    if set(split["train"]) & set(split["val"]):
        raise RuntimeError("Training and validation IDs overlap.")
    if not all(case.startswith("val_") for case in split["val"]):
        raise RuntimeError("Unexpected validation IDs.")

    run_dir = (
        project / "runs/nnunet" / dataset_name
        / f"seed_{args.seed}"
    )
    output = run_dir / "fold_0"

    snapshots = {
        "config.yaml": config_text,
        "plans_snapshot.json": plans_text,
        "split_snapshot.json": split_text,
    }

    if args.resume:
        for name, text in snapshots.items():
            path = run_dir / name
            if not path.exists() or path.read_text() != text:
                raise RuntimeError(
                    f"Resume configuration changed: {name}"
                )
        if (output / "checkpoint_final.safetensors").exists():
            raise RuntimeError("This run already finished.")
        latest = output / "checkpoint_latest.safetensors"
        if not latest.exists():
            raise FileNotFoundError(latest)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                f"Run folder is not empty: {run_dir}. "
                "Use --resume only for an unfinished run."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, text in snapshots.items():
            (run_dir / name).write_text(text)

    plans["continue_training"] = args.resume
    trainer = nnUNetTrainerPhase0FullVal(
        plans=plans,
        configuration=config["configuration"],
        fold=config["data"]["fold"],
        dataset_json=dataset_json,
        device=torch.device("cuda", 0),
    )

    # Override native output paths before initialization/training.
    output.mkdir(parents=True, exist_ok=True)
    trainer.output_folder_base = str(run_dir)
    trainer.output_folder = str(output)
    trainer.log_file = str(output / "training_log.txt")

    # Preserve native optimization, loss, augmentation and iteration counts.
    trainer.num_epochs = config["training"]["max_epochs"]
    trainer.save_every = 1
    trainer.early_stopping_patience = args.early_stopping_patience

    # Seed immediately before network initialization.
    set_seed(args.seed)
    trainer.initialize()

    if trainer.batch_size != config["training"]["batch_size"]:
        raise RuntimeError(
            f"Planned batch size {trainer.batch_size} differs from config."
        )

    if args.resume:
        trainer.load_checkpoint(str(latest))
        # Native checkpoints restore training state but do not capture
        # every augmentation worker's RNG state.
        set_seed(args.seed + trainer.current_epoch)

    print(
        f"Dataset={args.dataset}, seed={args.seed}, "
        f"train={len(split['train'])}, val={len(split['val'])}, "
        f"batch_size={trainer.batch_size}, "
        f"epochs={trainer.num_epochs}, "
        f"train_iterations_per_epoch={trainer.num_iterations_per_epoch}, "
        f"start_epoch={trainer.current_epoch}, "
        f"GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print(f"Output directory: {output}", flush=True)

    trainer.run_training()

    print(
        f"Training finished. See "
        f"{output / 'training_stop.json'} for the stop reason, "
        f"and {output / 'best_fullval.json'} for the best Dice.",
        flush=True,
    )


if __name__ == "__main__":
    main()