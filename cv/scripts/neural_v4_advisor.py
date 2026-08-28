#!/usr/bin/env python3
"""Live belief-aware recommendations from a separate v4 checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from durak_v4.model import BeliefActorCritic
from neural_v3_advisor import NeuralV3Advisor


class NeuralV4Advisor(NeuralV3Advisor):
    def __init__(
        self, checkpoint: Path, search_ms: int = 0,
        search_simulations: int = 128,
    ):
        super().__init__(
            checkpoint, search_ms, search_simulations,
            model_class=BeliefActorCritic,
        )
