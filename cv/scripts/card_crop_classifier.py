#!/usr/bin/env python3
"""Train and run a compact classifier on localized card crops."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = tuple(
    f"{rank}{suit}"
    for rank in ("6", "7", "8", "9", "10", "J", "Q", "K", "A")
    for suit in ("H", "D", "C", "S")
)


class CardNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, len(CLASS_NAMES))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class WeightedCardEnsemble(nn.Module):
    """Average classifier logits while keeping older class knowledge stable."""

    def __init__(self, state_dicts, weights) -> None:
        super().__init__()
        self.models = nn.ModuleList()
        for state_dict in state_dicts:
            model = CardNet()
            model.load_state_dict(state_dict)
            self.models.append(model)
        total = float(sum(weights))
        self.weights = tuple(float(weight) / total for weight in weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return sum(
            weight * model(inputs)
            for weight, model in zip(self.weights, self.models)
        )


def load_card_classifier(path: Path) -> nn.Module:
    checkpoint = torch.load(path, weights_only=True)
    if "models" in checkpoint:
        model = WeightedCardEnsemble(
            checkpoint["models"], checkpoint.get("weights", [1] * len(checkpoint["models"]))
        )
    else:
        model = CardNet()
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def read_samples(data: Path, split: str) -> list[tuple[Path, tuple[float, ...], int]]:
    samples = []
    for label_path in sorted((data / split / "labels").glob("*.txt")):
        image_path = next((data / split / "images").glob(f"{label_path.stem}.*"))
        for line in label_path.read_text().splitlines():
            fields = line.split()
            if fields:
                samples.append((image_path, tuple(map(float, fields[1:])), int(fields[0])))
    return samples


def extract_crop(image: np.ndarray, box: tuple[float, ...]) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy, bw, bh = box
    x1 = max(0, round((cx - bw / 2) * width))
    y1 = max(0, round((cy - bh / 2) * height))
    x2 = min(width, round((cx + bw / 2) * width))
    y2 = min(height, round((cy + bh / 2) * height))
    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (64, 96), interpolation=cv2.INTER_AREA)


class CropDataset(Dataset):
    def __init__(self, samples, augment: bool = False):
        self.items = []
        self.augment = augment
        cache = {}
        for image_path, box, target in samples:
            image = cache.setdefault(image_path, cv2.imread(str(image_path)))
            self.items.append((extract_crop(image, box), target))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image, target = self.items[index]
        image = image.copy()
        if self.augment:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-12, 12)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        return tensor, target


def evaluate(model, loader):
    model.eval()
    correct1 = correct3 = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            top3 = logits.topk(3, dim=1).indices
            correct1 += (top3[:, 0] == targets).sum().item()
            correct3 += (top3 == targets[:, None]).any(dim=1).sum().item()
            total += len(targets)
    return correct1 / total, correct3 / total


def train(args):
    train_samples = read_samples(args.data, "train")
    val_samples = read_samples(args.data, "val")
    hard_stems = set(filter(None, args.hard_frames.split(",")))
    if hard_stems:
        hard_samples = [sample for sample in train_samples if sample[0].stem in hard_stems]
        train_samples.extend(hard_samples * args.hard_repeat)
        print(f"hard samples: {len(hard_samples)} x {args.hard_repeat}", flush=True)
    train_loader = DataLoader(CropDataset(train_samples, True), batch_size=32, shuffle=True)
    val_loader = DataLoader(CropDataset(val_samples), batch_size=64)
    counts = torch.bincount(torch.tensor([sample[2] for sample in train_samples]), minlength=36).float()
    weights = counts.sum() / counts.clamp_min(1)
    weights /= weights.mean()
    model = CardNet()
    if args.initial:
        checkpoint = torch.load(args.initial, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else (2e-4 if args.initial else 2e-3)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        top1, top3 = evaluate(model, val_loader)
        if top1 >= best:
            best = top1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": CLASS_NAMES}, args.output)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch}: top1={top1:.1%} top3={top3:.1%}", flush=True)
    print(f"best top1={best:.1%}; saved={args.output}")


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project / "cv/data_verified")
    parser.add_argument("--output", type=Path, default=project / "cv/models/card_crop_classifier.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hard-frames", default="")
    parser.add_argument("--hard-repeat", type=int, default=12)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
