#!/usr/bin/env python3
"""Export the existing card recognizers to ONNX assets for Android."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

from card_crop_classifier import CLASS_NAMES, load_card_classifier


def export_localizer(source: Path, target: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="durak-onnx-") as directory:
        temporary = Path(directory) / "card_localizer.pt"
        shutil.copy2(source, temporary)
        exported = Path(YOLO(temporary).export(
            format="onnx",
            imgsz=416,
            batch=1,
            dynamic=False,
            simplify=False,
            nms=False,
            opset=17,
            device="cpu",
        ))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, target)


def export_classifier(source: Path, target: Path) -> None:
    model = load_card_classifier(source)
    example = torch.zeros((1, 3, 96, 64), dtype=torch.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        target,
        input_names=("cards",),
        output_names=("logits",),
        dynamic_axes={"cards": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def verify_model(path: Path, input_name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(path, providers=("CPUExecutionProvider",))
    output = session.run(None, {input_name: np.zeros(shape, dtype=np.float32)})[0]
    return tuple(output.shape)


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--localizer", type=Path,
        default=project / "cv/models/card_localizer.pt",
    )
    parser.add_argument(
        "--classifier", type=Path,
        default=project / "cv/models/card_crop_classifier_v10.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project / "android/app/src/main/assets/models",
    )
    args = parser.parse_args()

    localizer = args.output / "card_localizer.onnx"
    classifier = args.output / "card_classifier.onnx"
    export_localizer(args.localizer, localizer)
    export_classifier(args.classifier, classifier)

    localizer_shape = verify_model(localizer, "images", (1, 3, 416, 416))
    classifier_shape = verify_model(classifier, "cards", (2, 3, 96, 64))
    if localizer_shape != (1, 5, 3549):
        raise ValueError(f"unexpected localizer output: {localizer_shape}")
    if classifier_shape != (2, len(CLASS_NAMES)):
        raise ValueError(f"unexpected classifier output: {classifier_shape}")

    print(f"Exported {localizer.relative_to(project)} ({localizer.stat().st_size} bytes)")
    print(f"Exported {classifier.relative_to(project)} ({classifier.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
