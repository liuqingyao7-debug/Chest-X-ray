from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Project settings
# ============================================================

PROJECT = Path(
    "/mnt/cv_data/users/seven/language-guided-medical-segmentation"
)

SEEDS = [1001, 1002, 1003]

COLORS = {
    1001: "tab:blue",
    1002: "tab:orange",
    1003: "tab:green",
}

MODELS = {
    "unet": "U-Net",
    "transunet": "TransUNet",
    "nnunet": "nnU-Net",
}

DATASETS = {
    "QaTa-COV19": "Dataset501_QaTa",
    "MosMedDataPlus": "Dataset502_MosMed",
}

OUTPUT = PROJECT / "reports" / "training_curves"

if not (PROJECT / "runs").is_dir():
    raise FileNotFoundError(
        f"Cannot find the runs folder. Check PROJECT: {PROJECT}"
    )

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Read U-Net / TransUNet CSV histories
# ============================================================

def read_csv_history(run_dir):
    path = run_dir / "history.csv"

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None

    required = ["epoch", "train_loss", "val_loss"]

    if not set(required).issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {path}\n"
            f"Expected: {required}\n"
            f"Found: {list(df.columns)}"
        )

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Ignore incomplete or nonfinite records.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    df = (
        df.drop_duplicates("epoch", keep="last")
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 3. Read nnU-Net native training logs
# ============================================================

def read_nnunet_history(run_dir):
    folder = run_dir / "fold_0"

    fixed_log = folder / "training_log.txt"

    if fixed_log.exists():
        paths = [fixed_log]
    else:
        paths = sorted(
            folder.glob("training_log*.txt"),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )

    if not paths:
        return None

    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    epoch_pattern = re.compile(r"\bEpoch\s+(\d+)\s*$")
    train_pattern = re.compile(rf"\btrain_loss\s+({number})")
    val_pattern = re.compile(rf"\bval_loss\s+({number})")

    records = {}

    for path in paths:
        current_epoch = None
        values = {}

        # Read a snapshot; this also works while training is running.
        lines = path.read_text(errors="replace").splitlines()

        for line in lines:
            match = epoch_pattern.search(line)

            if match:
                current_epoch = int(match.group(1))
                values = {}
                continue

            if current_epoch is None:
                continue

            match = train_pattern.search(line)
            if match:
                values["train_loss"] = float(match.group(1))

            match = val_pattern.search(line)
            if match:
                values["val_loss"] = float(match.group(1))

            if "train_loss" in values and "val_loss" in values:
                if not all(np.isfinite(value) for value in values.values()):
                    continue

                # Repeated epochs after resume overwrite earlier records.
                records[current_epoch] = {
                    # nnU-Net starts at epoch 0; plot starting at 1.
                    "epoch": current_epoch + 1,
                    "train_loss": values["train_loss"],
                    "val_loss": values["val_loss"],
                }

    if not records:
        return None

    return (
        pd.DataFrame(records.values())
        .sort_values("epoch")
        .reset_index(drop=True)
    )


# ============================================================
# 4. Unified history loader
# ============================================================

def load_history(model, dataset, seed):
    if model == "nnunet":
        run_dir = (
            PROJECT
            / "runs"
            / "nnunet"
            / DATASETS[dataset]
            / f"seed_{seed}"
        )
        return read_nnunet_history(run_dir)

    run_dir = (
        PROJECT
        / "runs"
        / model
        / dataset
        / f"seed_{seed}"
    )

    return read_csv_history(run_dir)


# ============================================================
# 5. Plot one model and one dataset
# ============================================================

def plot_model_dataset(model, title, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    found = False

    for seed in SEEDS:
        df = load_history(model, dataset, seed)

        if df is None or df.empty:
            print(
                f"[Skipped] {title} / {dataset} / seed {seed}: "
                "no complete loss history"
            )
            continue

        found = True

        print(
            f"[Loaded] {title} / {dataset} / seed {seed}: "
            f"{len(df)} epochs, "
            f"last epoch = {int(df['epoch'].iloc[-1])}"
        )

        for ax, metric in zip(
            axes, ["train_loss", "val_loss"]
        ):
            ax.plot(
                df["epoch"],
                df[metric],
                color=COLORS[seed],
                label=f"Seed {seed}",
                linewidth=1.3,
                alpha=0.9,
            )

    if not found:
        plt.close(fig)
        return []

    for ax, subtitle in zip(
        axes, ["Training loss", "Validation loss"]
    ):
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_xlim(left=1)

    fig.suptitle(f"{title} — {dataset}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    folder = OUTPUT / model
    folder.mkdir(parents=True, exist_ok=True)

    saved = []

    for extension in ["png", "pdf"]:
        path = folder / f"{model}_{dataset}_loss.{extension}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        saved.append(path)
        print(f"[Saved] {path}")

    plt.show()
    plt.close(fig)

    return saved


# ============================================================
# 6. Generate all figures
# ============================================================

all_saved = []

for model, title in MODELS.items():
    for dataset in DATASETS:
        all_saved.extend(
            plot_model_dataset(model, title, dataset)
        )

print(f"\nFinished: {len(all_saved)} files saved.")
print(f"Output folder: {OUTPUT}")