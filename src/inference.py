import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda:0")


# ==================================================
# U-Net
# ==================================================

def load_unet(dataset, seed):

    run_dir = (
        PROJECT / "runs/unet"
        / dataset / f"seed_{seed}"
    )

    config = yaml.safe_load(
        (run_dir / "config.yaml").read_text()
    )

    model_path = PROJECT / config["source"]["model_file"]

    spec = importlib.util.spec_from_file_location(
        "lvit_unet", model_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.UNet(
        n_channels=3,
        n_classes=1
    ).to(DEVICE)

    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


@torch.inference_mode()
def predict_unet(model, images):

    images = images.to(DEVICE)

    probabilities = model(images)

    predictions = probabilities[:, 0] >= 0.5

    return predictions.cpu().numpy()


# ==================================================
# TransUNet
# ==================================================

def load_transunet(dataset, seed):

    run_dir = (
        PROJECT / "runs/transunet"
        / dataset / f"seed_{seed}"
    )

    config = yaml.safe_load(
        (run_dir / "config.yaml").read_text()
    )

    sys.path.insert(
        0, str(PROJECT / "external/TransUNet")
    )

    from networks.vit_seg_modeling import (
        VisionTransformer,
        CONFIGS
    )

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
        num_classes=config["model"]["num_classes"]
    ).to(DEVICE)

    checkpoint = torch.load(
        run_dir / "best.pt",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


@torch.inference_mode()
def predict_transunet(model, images):

    images = images.to(DEVICE)

    logits = model(images)

    probabilities = logits.softmax(dim=1)[:, 1]

    predictions = probabilities >= 0.5

    return predictions.cpu().numpy()


# ==================================================
# nnU-Net
# ==================================================

def load_nnunet(dataset, seed):

    os.environ["nnUNet_compile"] = "false"

    os.environ["nnUNet_raw"] = str(
        PROJECT / "datasets/nnunet/raw"
    )

    os.environ["nnUNet_preprocessed"] = str(
        PROJECT / "datasets/nnunet/preprocessed"
    )

    os.environ["nnUNet_results"] = str(
        PROJECT / "runs/nnunet"
    )

    sys.path.insert(
        0, str(PROJECT / "src/nnunet_custom")
    )

    from nnUNetTrainerPhase0FullVal import (
        nnUNetTrainerPhase0FullVal
    )

    dataset_names = {
        "QaTa-COV19": "Dataset501_QaTa",
        "MosMedDataPlus": "Dataset502_MosMed"
    }

    dataset_name = dataset_names[dataset]

    run_dir = (
        PROJECT / "runs/nnunet"
        / dataset_name / f"seed_{seed}"
    )

    config = yaml.safe_load(
        (run_dir / "config.yaml").read_text()
    )

    plans = json.loads(
        (run_dir / "plans_snapshot.json").read_text()
    )

    plans["continue_training"] = False

    dataset_json = json.loads(
        (
            PROJECT / "datasets/nnunet/preprocessed"
            / dataset_name / "dataset.json"
        ).read_text()
    )

    trainer = nnUNetTrainerPhase0FullVal(
        plans=plans,
        configuration=config["configuration"],
        fold=config["data"]["fold"],
        dataset_json=dataset_json,
        device=DEVICE
    )

    trainer.initialize()

    checkpoint = (
        run_dir / "fold_0"
        / "checkpoint_best_fullval.safetensors"
    )

    trainer.load_checkpoint(str(checkpoint))
    trainer.network.eval()

    return trainer


@torch.inference_mode()
def predict_nnunet(trainer, image_files):

    # Native nnU-Net preprocessing
    preprocessor = (
        trainer.configuration_manager.preprocessor_class(
            verbose=False
        )
    )

    data, _, properties = preprocessor.run_case(
        image_files,
        None,
        trainer.plans_manager,
        trainer.configuration_manager,
        trainer.dataset_json
    )

    before = tuple(
        properties["shape_before_cropping"]
    )

    cropped = tuple(
        properties[
            "shape_after_cropping_and_before_resampling"
        ]
    )

    bbox = properties["bbox_used_for_cropping"]

    array = np.asarray(data, dtype=np.float32)

    assert before == (1, 224, 224)
    assert tuple(array.shape[1:]) == cropped
    assert array.shape[0] == 3
    assert array.shape[1] == 1

    height, width = array.shape[-2:]

    assert 0 < height <= 224
    assert 0 < width <= 224

    top = (224 - height) // 2
    left = (224 - width) // 2

    image = torch.from_numpy(
        np.ascontiguousarray(array[:, 0])
    )[None].to(DEVICE)

    # Pad to 224 × 224
    image = F.pad(
        image,
        (
            left,
            224 - width - left,
            top,
            224 - height - top
        )
    )

    # Network prediction
    with torch.autocast(
        DEVICE.type,
        enabled=DEVICE.type == "cuda"
    ):
        output = trainer.network(image)

    if isinstance(output, (list, tuple)):
        output = output[0]

    assert tuple(output.shape) == (1, 2, 224, 224)
    assert torch.isfinite(output).all()

    probability = output.float().softmax(dim=1)[0, 1]

    # Remove padding
    prediction = (
        probability[
            top:top + height,
            left:left + width
        ] >= 0.5
    ).cpu().numpy()

    # Restore to the common evaluation space
    restored = np.zeros((224, 224), dtype=bool)

    y0, y1 = bbox[1]
    x0, x1 = bbox[2]

    assert prediction.shape == (y1 - y0, x1 - x0)

    restored[y0:y1, x0:x1] = prediction

    return restored