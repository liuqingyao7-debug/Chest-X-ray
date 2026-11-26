from pathlib import Path
import csv
import cv2
import numpy as np

DATASETS = {
    "MosMedDataPlus": (
        Path("datasets/MosMedDataPlus"),
        {"Train_Folder": 2183, "Val_Folder": 273, "Test_Folder": 273},
    ),
    "QaTa-COV19": (
        Path("datasets/QaTa-Covid19"),
        {"Train_Folder": 5716, "Val_Folder": 1429, "Test_Folder": 2113},
    ),
}
SPLITS = {
    "Train_Folder": "train",
    "Val_Folder": "val",
    "Test_Folder": "test",
}
OUTPUT = Path("datasets/processed")
rows = []

for dataset, (root, counts) in DATASETS.items():
    split_ids = {}

    # Check counts, pairing, and split overlap before processing.
    for folder, expected in counts.items():
        images = {p.name for p in (root / folder / "img").glob("*.png")}
        masks = {p.name for p in (root / folder / "labelcol").glob("*.png")}
        if images != masks:
            raise RuntimeError(f"{dataset}/{folder}: unmatched filenames")
        if len(images) != expected:
            raise RuntimeError(
                f"{dataset}/{folder}: expected {expected}, found {len(images)}"
            )
        split_ids[folder] = images

    folders = list(split_ids)
    for i, first in enumerate(folders):
        for second in folders[i + 1:]:
            overlap = split_ids[first] & split_ids[second]
            if overlap:
                raise RuntimeError(
                    f"{dataset}: {len(overlap)} overlapping IDs "
                    f"between {first} and {second}"
                )

    for folder, expected in counts.items():
        split = SPLITS[folder]
        image_out = OUTPUT / dataset / split / "images"
        mask_out = OUTPUT / dataset / split / "masks"
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        empty = 0

        for index, filename in enumerate(sorted(split_ids[folder]), 1):
            image_src = root / folder / "img" / filename
            mask_src = root / folder / "labelcol" / filename

            image = cv2.imread(str(image_src))
            mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise RuntimeError(f"Unreadable pair: {image_src}")

            # Match the verified LViT loading and resizing order.
            image = cv2.resize(image, (224, 224))
            mask = cv2.resize(mask, (224, 224))
            mask = (mask > 0).astype(np.uint8)
            empty += int(not mask.any())

            image_dst = image_out / filename
            mask_dst = mask_out / filename

            if not cv2.imwrite(str(image_dst), image):
                raise RuntimeError(f"Cannot save {image_dst}")
            if not cv2.imwrite(str(mask_dst), mask):
                raise RuntimeError(f"Cannot save {mask_dst}")

            # Read back and verify the actual saved files.
            saved_image = cv2.imread(str(image_dst))
            saved_mask = cv2.imread(str(mask_dst), cv2.IMREAD_UNCHANGED)
            if saved_image is None or saved_mask is None:
                raise RuntimeError(f"Unreadable output: {filename}")
            if saved_image.shape != (224, 224, 3):
                raise RuntimeError(f"Unexpected image shape: {image_dst}")
            if saved_mask.shape != (224, 224):
                raise RuntimeError(f"Unexpected mask shape: {mask_dst}")
            if saved_mask.dtype != np.uint8:
                raise RuntimeError(f"Unexpected mask dtype: {mask_dst}")
            if not set(np.unique(saved_mask)).issubset({0, 1}):
                raise RuntimeError(f"Nonbinary mask: {mask_dst}")
            if not np.array_equal(saved_mask, mask):
                raise RuntimeError(f"Saved mask changed: {mask_dst}")

            rows.append({
                "dataset": dataset,
                "split": split,
                "sample_id": Path(filename).stem,
                "source_image": str(image_src),
                "source_mask": str(mask_src),
                "prepared_image": str(image_dst),
                "prepared_mask": str(mask_dst),
            })

            if index % 500 == 0:
                print(f"{dataset}/{split}: {index}/{expected}", flush=True)

        expected_names = split_ids[folder]
        for directory in (image_out, mask_out):
            actual_names = {p.name for p in directory.iterdir() if p.is_file()}
            if actual_names != expected_names:
                raise RuntimeError(f"Unexpected output files in {directory}")

        print(
            f"PASS {dataset}/{split}: {expected} pairs, "
            f"224x224, binary masks, empty masks={empty}",
            flush=True,
        )

manifest = OUTPUT / "phase0_manifest.csv"
with manifest.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCOMPLETE: {len(rows)} verified pairs", flush=True)
print(f"Manifest: {manifest}", flush=True)
