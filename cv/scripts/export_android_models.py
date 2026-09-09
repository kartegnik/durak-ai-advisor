#!/usr/bin/env python3
"""Export the existing card recognizers to ONNX assets for Android."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from card_crop_classifier import CLASS_NAMES, load_card_classifier
from durak_v3.model import RecurrentActorCritic
from durak_v3.observation import OBSERVATION_DIM
from durak_v4.model import BeliefActorCritic
from model import DurakNet
from state import COMBINED_DIM, OPTION_DIM


class PolicyStep(torch.nn.Module):
    """Expose only the recurrent policy path used by the live advisor."""

    def __init__(self, model: RecurrentActorCritic):
        super().__init__()
        self.model = model

    def forward(self, observation, options, hidden):
        encoded = self.model.observation_encoder(observation)
        next_hidden = self.model.memory(encoded, hidden)
        option_features = self.model.option_encoder(options)
        context = next_hidden.unsqueeze(0).expand(option_features.shape[0], -1)
        logits = self.model.policy(
            torch.cat((context, option_features), dim=-1),
        ).squeeze(-1)
        return logits, next_hidden


class BeliefPolicyStep(torch.nn.Module):
    """Expose the two-memory v4 policy without its training-only value head."""

    def __init__(self, model: BeliefActorCritic):
        super().__init__()
        self.model = model

    def forward(self, observation, options, policy_hidden, belief_hidden):
        encoded = self.model.observation_encoder(observation)
        next_policy_hidden = self.model.memory(encoded, policy_hidden)
        belief_encoded = self.model.belief_observation_encoder(observation)
        next_belief_hidden = self.model.belief_memory(
            belief_encoded, belief_hidden,
        )
        option_features = self.model.option_encoder(options)
        policy_context = self.model.policy_context(
            next_policy_hidden, next_belief_hidden,
        )
        context = policy_context.unsqueeze(0).expand(option_features.shape[0], -1)
        logits = self.model.policy(
            torch.cat((context, option_features), dim=-1),
        ).squeeze(-1)
        return logits, next_policy_hidden, next_belief_hidden


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


def export_v1_policy(source: Path, target: Path) -> None:
    """Export the original 5,000-episode policy with transfer compatibility."""
    saved = torch.load(source, map_location="cpu", weights_only=True)
    model = DurakNet()
    model.load_compatible_state_dict(saved.get("model", saved))
    example = torch.zeros((2, COMBINED_DIM), dtype=torch.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.eval(),
        example,
        target,
        input_names=("combined",),
        output_names=("scores",),
        dynamic_axes={"combined": {0: "options"}, "scores": {0: "options"}},
        opset_version=17,
        dynamo=False,
    )


def export_v3_policy(source: Path, target: Path) -> int:
    saved = torch.load(source, map_location="cpu", weights_only=True)
    model = RecurrentActorCritic(**saved.get("model_config", {}))
    model.load_state_dict(saved["model"])
    wrapper = PolicyStep(model.eval())
    observation = torch.zeros(OBSERVATION_DIM, dtype=torch.float32)
    options = torch.zeros((2, OPTION_DIM), dtype=torch.float32)
    hidden = model.initial_hidden()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (observation, options, hidden),
        target,
        input_names=("observation", "options", "hidden"),
        output_names=("logits", "next_hidden"),
        dynamic_axes={"options": {0: "options"}, "logits": {0: "options"}},
        opset_version=17,
        dynamo=False,
    )
    return model.hidden_size


def export_v4_policy(source: Path, target: Path) -> int:
    saved = torch.load(source, map_location="cpu", weights_only=True)
    model = BeliefActorCritic(**saved.get("model_config", {}))
    model.load_state_dict(saved["model"])
    wrapper = BeliefPolicyStep(model.eval())
    observation = torch.zeros(OBSERVATION_DIM, dtype=torch.float32)
    options = torch.zeros((2, OPTION_DIM), dtype=torch.float32)
    policy_hidden = model.initial_hidden()
    belief_hidden = model.initial_belief_hidden()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (observation, options, policy_hidden, belief_hidden),
        target,
        input_names=(
            "observation", "options", "policy_hidden", "belief_hidden",
        ),
        output_names=(
            "logits", "next_policy_hidden", "next_belief_hidden",
        ),
        dynamic_axes={"options": {0: "options"}, "logits": {0: "options"}},
        opset_version=17,
        dynamo=False,
    )
    return model.hidden_size


def verify_model(path: Path, inputs: dict[str, np.ndarray]) -> tuple[tuple[int, ...], ...]:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(path, providers=("CPUExecutionProvider",))
    outputs = session.run(None, inputs)
    return tuple(tuple(output.shape) for output in outputs)


def verify_v4_equivalence(source: Path, target: Path) -> None:
    """Check that export keeps v4 scores and both memories numerically intact."""
    saved = torch.load(source, map_location="cpu", weights_only=True)
    model = BeliefActorCritic(**saved.get("model_config", {}))
    model.load_state_dict(saved["model"])
    wrapper = BeliefPolicyStep(model.eval())
    generator = torch.Generator().manual_seed(20260909)
    inputs = (
        torch.randn(OBSERVATION_DIM, generator=generator),
        torch.randn(5, OPTION_DIM, generator=generator),
        torch.randn(model.hidden_size, generator=generator),
        torch.randn(model.hidden_size, generator=generator),
    )
    with torch.no_grad():
        expected = tuple(value.numpy() for value in wrapper(*inputs))
    session = ort.InferenceSession(target, providers=("CPUExecutionProvider",))
    actual = session.run(None, {
        "observation": inputs[0].numpy(),
        "options": inputs[1].numpy(),
        "policy_hidden": inputs[2].numpy(),
        "belief_hidden": inputs[3].numpy(),
    })
    for name, expected_value, actual_value in zip(
            ("logits", "policy hidden", "belief hidden"), expected, actual):
        if not np.allclose(expected_value, actual_value, rtol=1e-5, atol=1e-5):
            difference = np.max(np.abs(expected_value - actual_value))
            raise ValueError(f"v4 {name} differs after ONNX export: {difference}")


def main() -> None:
    project = PROJECT
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
        "--policy-v1", type=Path,
        default=project / "durak_model.pt",
    )
    parser.add_argument(
        "--policy-v3", type=Path,
        default=project / "durak_model_v3.pt",
    )
    parser.add_argument(
        "--policy-v4", type=Path,
        default=project / "durak_model_v4.pt",
    )
    parser.add_argument(
        "--only-policy-v4", action="store_true",
        help="Skip unchanged card, v1 and v3 exports.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project / "android/app/src/main/assets/models",
    )
    args = parser.parse_args()

    localizer = args.output / "card_localizer.onnx"
    classifier = args.output / "card_classifier.onnx"
    policy_v1 = args.output / "durak_policy_v1.onnx"
    policy_v3 = args.output / "durak_policy_v3.onnx"
    policy_v4 = args.output / "durak_policy_v4.onnx"
    if not args.only_policy_v4:
        export_localizer(args.localizer, localizer)
        export_classifier(args.classifier, classifier)
        export_v1_policy(args.policy_v1, policy_v1)
        policy_hidden_size = export_v3_policy(args.policy_v3, policy_v3)
    belief_hidden_size = export_v4_policy(args.policy_v4, policy_v4)

    if not args.only_policy_v4:
        localizer_shapes = verify_model(localizer, {
            "images": np.zeros((1, 3, 416, 416), dtype=np.float32),
        })
        classifier_shapes = verify_model(classifier, {
            "cards": np.zeros((2, 3, 96, 64), dtype=np.float32),
        })
        policy_v1_shapes = verify_model(policy_v1, {
            "combined": np.zeros((2, COMBINED_DIM), dtype=np.float32),
        })
        policy_v3_shapes = verify_model(policy_v3, {
            "observation": np.zeros(OBSERVATION_DIM, dtype=np.float32),
            "options": np.zeros((2, OPTION_DIM), dtype=np.float32),
            "hidden": np.zeros(policy_hidden_size, dtype=np.float32),
        })
        if localizer_shapes != ((1, 5, 3549),):
            raise ValueError(f"unexpected localizer output: {localizer_shapes}")
        if classifier_shapes != ((2, len(CLASS_NAMES)),):
            raise ValueError(f"unexpected classifier output: {classifier_shapes}")
        if policy_v1_shapes != ((2,),):
            raise ValueError(f"unexpected v1 policy output: {policy_v1_shapes}")
        if policy_v3_shapes != ((2,), (policy_hidden_size,)):
            raise ValueError(f"unexpected v3 policy output: {policy_v3_shapes}")
    policy_v4_shapes = verify_model(policy_v4, {
        "observation": np.zeros(OBSERVATION_DIM, dtype=np.float32),
        "options": np.zeros((2, OPTION_DIM), dtype=np.float32),
        "policy_hidden": np.zeros(belief_hidden_size, dtype=np.float32),
        "belief_hidden": np.zeros(belief_hidden_size, dtype=np.float32),
    })
    if policy_v4_shapes != (
        (2,), (belief_hidden_size,), (belief_hidden_size,),
    ):
        raise ValueError(f"unexpected v4 policy output: {policy_v4_shapes}")
    verify_v4_equivalence(args.policy_v4, policy_v4)

    if not args.only_policy_v4:
        print(f"Exported {localizer.relative_to(project)} ({localizer.stat().st_size} bytes)")
        print(f"Exported {classifier.relative_to(project)} ({classifier.stat().st_size} bytes)")
        print(f"Exported {policy_v1.relative_to(project)} ({policy_v1.stat().st_size} bytes)")
        print(f"Exported {policy_v3.relative_to(project)} ({policy_v3.stat().st_size} bytes)")
    print(f"Exported {policy_v4.relative_to(project)} ({policy_v4.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
