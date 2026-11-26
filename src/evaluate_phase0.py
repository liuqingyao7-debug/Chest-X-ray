import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from phase0_dataset import Phase0Dataset
from metrics import compute_metrics
from inference import (
    load_unet,
    predict_unet,
    load_transunet,
    predict_transunet,
    load_nnunet,
    predict_nnunet,
)


PROJECT = Path(__file__).resolve().parents[1]

EXPECTED_COUNTS = {
    "QaTa-COV19": 2113,
    "MosMedDataPlus": 273,
}

DATASET_NAMES = {
    "QaTa-COV19": "Dataset501_QaTa",
    "MosMedDataPlus": "Dataset502_MosMed",
}

METRICS = ["dice", "miou", "hd95", "assd"]


def save_result(sample_id, prediction, target, output, rows):

    assert prediction.shape == (224, 224)
    assert target.shape == (224, 224)

    scores = compute_metrics(prediction, target)

    saved = cv2.imwrite(
        str(output / "predicted_masks" / f"{sample_id}.png"),
        prediction.astype(np.uint8) * 255,
    )

    if not saved:
        raise RuntimeError(f"Failed to save mask: {sample_id}")

    rows.append({
        "sample_id": sample_id,
        **scores,
    })


def evaluate_unet_transunet(args, output, rows):

    run_dir = (
        PROJECT / "runs" / args.model
        / args.dataset / f"seed_{args.seed}"
    )

    config = yaml.safe_load(
        (run_dir / "config.yaml").read_text()
    )

    dataset = Phase0Dataset(
        config["data"]["root"],
        args.dataset,
        "test",
        augment=False,
    )

    assert len(dataset) == EXPECTED_COUNTS[args.dataset]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.model == "unet":
        model = load_unet(args.dataset, args.seed)
        predict = predict_unet

    else:
        model = load_transunet(args.dataset, args.seed)
        predict = predict_transunet

    for batch in loader:

        predictions = predict(model, batch["image"])
        targets = batch["mask"][:, 0].numpy()

        for index, sample_id in enumerate(batch["sample_id"]):

            save_result(
                sample_id,
                predictions[index],
                targets[index],
                output,
                rows,
            )

            if len(rows) % 100 == 0:
                print(
                    f"Evaluated {len(rows)}/{len(dataset)}",
                    flush=True,
                )

            if args.limit and len(rows) >= args.limit:
                return


def evaluate_nnunet(args, output, rows):

    dataset_name = DATASET_NAMES[args.dataset]

    raw_folder = (
        PROJECT / "datasets/nnunet/raw"
        / dataset_name
    )

    with (raw_folder / "case_mapping.csv").open() as file:

        mapping = [
            row for row in csv.DictReader(file)
            if row["split"] == "test"
        ]

    assert len(mapping) == EXPECTED_COUNTS[args.dataset]

    sample_ids = [row["original_sample_id"] for row in mapping]
    assert len(set(sample_ids)) == len(mapping)

    trainer = load_nnunet(args.dataset, args.seed)

    for row in mapping:

        case_id = row["case_id"]
        sample_id = row["original_sample_id"]

        image_files = [
            str(
                raw_folder / "imagesTs"
                / f"{case_id}_{channel:04d}.png"
            )
            for channel in range(3)
        ]

        prediction = predict_nnunet(
            trainer, image_files
        )

        target = cv2.imread(
            str(PROJECT / row["prepared_mask"]),
            cv2.IMREAD_UNCHANGED,
        )

        if target is None:
            raise RuntimeError(
                f"Cannot read target: {sample_id}"
            )

        assert target.shape == (224, 224)
        assert np.isin(target, [0, 1]).all()

        save_result(
            sample_id,
            prediction,
            target,
            output,
            rows,
        )

        if len(rows) % 100 == 0:
            print(
                f"Evaluated {len(rows)}/{len(mapping)}",
                flush=True,
            )

        if args.limit and len(rows) >= args.limit:
            return


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["unet", "transunet", "nnunet"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["QaTa-COV19", "MosMedDataPlus"],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[1001, 1002, 1003],
    )

    parser.add_argument(
        "--batch-size", type=int, default=4
    )

    parser.add_argument(
        "--workers", type=int, default=2
    )

    parser.add_argument(
        "--limit", type=int, default=None
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    os.chdir(PROJECT)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Run evaluation inside a GPU allocation."
        )

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Keep smoke-test results separate from final results.
    folder = (
        "phase0_smoke"
        if args.limit is not None
        else "phase0_evaluation"
    )

    output = (
        PROJECT / "results" / folder
        / args.model
        / args.dataset
        / f"seed_{args.seed}"
    )

    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output}"
        )

    (output / "predicted_masks").mkdir(
        parents=True, exist_ok=True
    )

    rows = []

    if args.model == "nnunet":
        evaluate_nnunet(args, output, rows)
    else:
        evaluate_unet_transunet(args, output, rows)

    expected = EXPECTED_COUNTS[args.dataset]

    if args.limit is not None:
        expected = min(expected, args.limit)

    assert len(rows) == expected
    assert len({r["sample_id"] for r in rows}) == expected

    # Save per-case metrics.
    with (output / "per_case_metrics.csv").open(
        "w", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id"] + METRICS,
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save test summary.
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "num_cases": len(rows),
        "smoke_test": args.limit is not None,
        "evaluation_size": [224, 224],
        "threshold": 0.5,
        "iou_definition": "foreground_only",
        "distance_unit": "pixels",
        "empty_mask_penalty": float(np.hypot(223, 223)),
    }

    for metric in METRICS:

        values = [row[metric] for row in rows]

        summary[f"mean_{metric}"] = float(
            np.mean(values)
        )

    (output / "test_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print(f"Evaluation complete: {output}")


if __name__ == "__main__":
    main()