import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerPhase0FullVal(nnUNetTrainer):
    @torch.no_grad()
    def evaluate_full_validation(self):
        if self.is_ddp:
            raise RuntimeError("This benchmark wrapper supports one GPU per run.")

        folder = Path(self.preprocessed_dataset_folder)
        split_path = Path(self.preprocessed_dataset_folder_base) / "splits_final.json"
        splits = json.loads(split_path.read_text())
        assert self.fold == 0 and len(splits) == 1
        ids = splits[0]["val"]
        assert ids and all(case.startswith("val_") for case in ids)

        dataset_class = infer_dataset_class(str(folder))
        dataset = dataset_class(str(folder), ids)
        labels = (
            Path(str(nnUNet_raw))
            / self.plans_manager.dataset_name / "labelsTr"
        )
        was_training = self.network.training
        self.network.eval()
        scores = []

        try:
            for index, case_id in enumerate(ids, 1):
                data, _, previous = dataset.load_case(case_id)
                assert previous is None

                with (folder / f"{case_id}.pkl").open("rb") as file:
                    properties = pickle.load(file)

                before = tuple(properties["shape_before_cropping"])
                cropped = tuple(
                    properties["shape_after_cropping_and_before_resampling"]
                )
                bbox = properties["bbox_used_for_cropping"]
                array = np.asarray(data[:], dtype=np.float32)

                # Require unchanged 2D spacing/size after cropping.
                assert before == (1, 224, 224), (case_id, before)
                assert tuple(array.shape[1:]) == cropped, case_id
                assert array.shape[0] == 3 and array.shape[1] == 1
                assert bbox[0] == [0, 1] or tuple(bbox[0]) == (0, 1)

                height, width = array.shape[-2:]
                assert 0 < height <= 224 and 0 < width <= 224
                top = (224 - height) // 2
                left = (224 - width) // 2

                image = torch.from_numpy(
                    np.ascontiguousarray(array[:, 0])
                )[None].to(self.device)
                image = F.pad(
                    image,
                    (left, 224 - width - left,
                     top, 224 - height - top),
                )

                with torch.autocast(
                    self.device.type, enabled=self.device.type == "cuda"
                ):
                    output = self.network(image)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                assert tuple(output.shape) == (1, 2, 224, 224)
                assert torch.isfinite(output).all()

                probability = output.float().softmax(1)[0, 1]
                prediction = (
                    probability[top:top + height, left:left + width] >= 0.5
                ).cpu().numpy()

                restored = np.zeros((224, 224), dtype=bool)
                y0, y1 = bbox[1]
                x0, x1 = bbox[2]
                assert prediction.shape == (y1 - y0, x1 - x0)
                restored[y0:y1, x0:x1] = prediction

                with Image.open(labels / f"{case_id}.png") as image_file:
                    target = np.asarray(image_file)
                assert target.shape == (224, 224)
                assert np.isin(target, [0, 1]).all()
                target = target.astype(bool)

                denominator = int(restored.sum()) + int(target.sum())
                dice = (
                    1.0 if denominator == 0 else
                    2.0 * np.logical_and(restored, target).sum() / denominator
                )
                scores.append(float(dice))

                if index % 250 == 0:
                    print(f"Full validation: {index}/{len(ids)}", flush=True)
        finally:
            self.network.train(was_training)

        return {
            "mean_dice": float(np.mean(scores)),
            "num_cases": len(scores),
            "epoch": int(self.current_epoch),
        }

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        result = self.evaluate_full_validation()
        self.print_to_log_file("Full-image validation Dice:", result["mean_dice"])

        destination = Path(self.output_folder)
        metadata = destination / "best_fullval.json"
        best = (
            json.loads(metadata.read_text())["mean_dice"]
            if metadata.exists() else -1.0
        )

        with (destination / "fullval_history.jsonl").open("a") as file:
            file.write(json.dumps(result) + "\n")

        if result["mean_dice"] > best:
            self.save_checkpoint(
                str(destination / "checkpoint_best_fullval.safetensors")
            )
            metadata.write_text(json.dumps(result, indent=2) + "\n")
        
        self.update_early_stopping(result["mean_dice"])


    def restore_early_stopping(self):
        """Reconstruct the counter from completed checkpoint epochs."""
        self.early_stopping_best = -float("inf")
        self.early_stopping_bad_epochs = 0

        history_path = Path(self.output_folder) / "fullval_history.jsonl"
        if history_path.exists():
            # A resumed checkpoint starts at current_epoch.
            # Ignore validation records from epochs not in that checkpoint.
            records = {}
            for line in history_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                epoch = int(record["epoch"])
                if epoch < int(self.current_epoch):
                    records[epoch] = record

            # Repeated epochs after interruption count only once.
            for epoch in sorted(records):
                score = float(records[epoch]["mean_dice"])
                if not np.isfinite(score):
                    raise RuntimeError(
                        f"Nonfinite full-validation Dice at epoch {epoch}"
                    )

                if score > self.early_stopping_best:
                    self.early_stopping_best = score
                    self.early_stopping_bad_epochs = 0
                else:
                    self.early_stopping_bad_epochs += 1

            # Do not silently reset patience if resume history is incomplete.
            expected = set(range(int(self.current_epoch)))
            if set(records) != expected:
                raise RuntimeError(
                    "Full-validation history is incomplete. "
                    "Restore fullval_history.jsonl before resuming "
                    "with early stopping."
                )

        elif int(self.current_epoch) > 0:
            raise RuntimeError(
                "Missing fullval_history.jsonl; cannot restore "
                "the early-stopping counter."
            )

        self.print_to_log_file(
            "Early stopping restored:",
            f"patience={self.early_stopping_patience},",
            f"bad_epochs={self.early_stopping_bad_epochs},",
            f"best_fullval_dice={self.early_stopping_best}",
        )

    def update_early_stopping(self, score):
        """Count consecutive epochs without a strict Dice improvement."""
        score = float(score)
        if not np.isfinite(score):
            raise RuntimeError("Nonfinite full-validation Dice.")

        if score > self.early_stopping_best:
            self.early_stopping_best = score
            self.early_stopping_bad_epochs = 0
        else:
            self.early_stopping_bad_epochs += 1

        self.print_to_log_file(
            "Early stopping:",
            f"{self.early_stopping_bad_epochs}/"
            f"{self.early_stopping_patience}",
        )

    def run_training(self):
        """Preserve the native training steps and add an epoch-end stop."""
        if not hasattr(self, "early_stopping_patience"):
            raise RuntimeError("Set early_stopping_patience before training.")

        self.restore_early_stopping()

        # Record the actual stopping policy without changing config.yaml.
        policy = {
            "enabled": True,
            "metric": "mean_per_case_foreground_dice",
            "patience": self.early_stopping_patience,
            "min_delta": 0.0,
            "max_epochs": self.num_epochs,
        }
        policy_path = Path(self.output_folder) / "early_stopping_policy.json"
        if policy_path.exists():
            previous = json.loads(policy_path.read_text())
            if previous != policy:
                raise RuntimeError(
                    "Early-stopping policy changed within this run."
                )
        else:
            policy_path.write_text(json.dumps(policy, indent=2) + "\n")

        self.on_train_start()

        while self.current_epoch < self.num_epochs:
            # Also check restored patience before starting another epoch.
            if (
                self.early_stopping_bad_epochs
                >= self.early_stopping_patience
            ):
                break

            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(
                    self.train_step(next(self.dataloader_train))
                )
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(
                        self.validation_step(next(self.dataloader_val))
                    )
                self.on_validation_epoch_end(val_outputs)

            # Keep native logging, checkpointing and epoch increment.
            self.on_epoch_end()

        stopped_early = (
            self.current_epoch < self.num_epochs
            and self.early_stopping_bad_epochs
            >= self.early_stopping_patience
        )

        summary = {
            "reason": (
                "early_stopping" if stopped_early
                else "max_epochs_reached"
            ),
            "completed_epochs": int(self.current_epoch),
            "best_fullval_dice": float(self.early_stopping_best),
            "bad_epochs": int(self.early_stopping_bad_epochs),
            "patience": int(self.early_stopping_patience),
        }

        destination = Path(self.output_folder) / "training_stop.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(temporary, destination)

        self.print_to_log_file(
            "Training stop reason:", summary["reason"],
            "| completed epochs:", summary["completed_epochs"],
        )

        # Save final weights and shut down data loaders normally.
        self.on_train_end()