from pathlib import Path
import csv
import json
import shutil

import cv2
import numpy as np

prepared = Path("datasets/processed")
raw = Path("datasets/nnunet/raw")
split_output = Path("configs/nnunet_splits")
split_output.mkdir(parents=True, exist_ok=True)

datasets = [
    ("QaTa-COV19", "Dataset501_QaTa", (5716, 1429, 2113)),
    ("MosMedDataPlus", "Dataset502_MosMed", (2183, 273, 273)),
]

for source_name, target_name, counts in datasets:
    destination = raw / target_name
    if destination.exists():
        raise RuntimeError(
            f"{destination} already exists. Inspect it before rerunning."
        )

    for folder in ["imagesTr", "labelsTr", "imagesTs"]:
        (destination / folder).mkdir(parents=True, exist_ok=True)

    split_ids = {}
    mapping = []
    seen = set()

    for split, expected in zip(["train", "val", "test"], counts):
        source = prepared / source_name / split
        images = sorted((source / "images").glob("*.png"))
        assert len(images) == expected, (source_name, split, len(images))
        split_ids[split] = []

        for index, image_path in enumerate(images):
            assert image_path.stem not in seen, image_path.stem
            seen.add(image_path.stem)

            case_id = f"{split}_{index:05d}"
            split_ids[split].append(case_id)

            image = cv2.imread(str(image_path))
            mask_path = source / "masks" / image_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

            assert image is not None and image.shape == (224, 224, 3)
            assert mask is not None and mask.shape == (224, 224)
            assert np.isin(mask, [0, 1]).all()

            image_folder = "imagesTs" if split == "test" else "imagesTr"

            # Store each existing image channel as one grayscale PNG.
            # This preserves the prepared channel order and pixel values.
            for channel in range(3):
                output = (
                    destination / image_folder
                    / f"{case_id}_{channel:04d}.png"
                )
                assert cv2.imwrite(str(output), image[:, :, channel])
                saved = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
                assert np.array_equal(saved, image[:, :, channel])

            if split != "test":
                output = destination / "labelsTr" / f"{case_id}.png"
                shutil.copyfile(mask_path, output)
                saved = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
                assert np.array_equal(saved, mask)

            mapping.append({
                "case_id": case_id,
                "split": split,
                "original_sample_id": image_path.stem,
                "prepared_image": str(image_path),
                "prepared_mask": str(mask_path),
            })

        print(f"PASS {source_name}/{split}: {expected} cases", flush=True)

    description = {
        "channel_names": {
            "0": "channel_0",
            "1": "channel_1",
            "2": "channel_2",
        },
        "labels": {"background": 0, "lesion": 1},
        "numTraining": counts[0] + counts[1],
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (destination / "dataset.json").write_text(
        json.dumps(description, indent=2) + "\n"
    )

    # Stage the fixed split; we will install it into the preprocessed
    # dataset directory before invoking any trainer.
    fixed_split = [{
        "train": split_ids["train"],
        "val": split_ids["val"],
    }]
    (split_output / f"{target_name}.json").write_text(
        json.dumps(fixed_split, indent=2) + "\n"
    )

    with (destination / "case_mapping.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)

    print(f"READY: {destination}", flush=True)

print("\nConversion complete. No training started.")
