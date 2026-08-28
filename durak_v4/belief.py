"""Losses and diagnostics for hidden-card prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from state import NUM_CARDS
from .model import DECK, OPPONENT


KNOWN_OPPONENT_OFFSET = NUM_CARDS * 4
UNKNOWN_OFFSET = NUM_CARDS * 5


def belief_targets(privileged: torch.Tensor) -> torch.Tensor:
    """Map the privileged opponent/deck masks to one class per card."""
    targets = torch.zeros(privileged.shape[:-1] + (NUM_CARDS,), dtype=torch.long,
                          device=privileged.device)
    targets = targets.masked_fill(privileged[..., :NUM_CARDS] > 0.5, OPPONENT)
    targets = targets.masked_fill(privileged[..., NUM_CARDS:] > 0.5, DECK)
    return targets


def candidate_mask(observation: torch.Tensor, include_known: bool = True) -> torch.Tensor:
    """Cards that can still be in the opponent hand or deck."""
    unknown = observation[..., UNKNOWN_OFFSET:UNKNOWN_OFFSET + NUM_CARDS] > 0.5
    if not include_known:
        return unknown
    known = observation[
        ..., KNOWN_OPPONENT_OFFSET:KNOWN_OPPONENT_OFFSET + NUM_CARDS
    ] > 0.5
    return unknown | known


def belief_cross_entropy(
    logits: torch.Tensor,
    observation: torch.Tensor,
    privileged: torch.Tensor,
) -> torch.Tensor:
    mask = candidate_mask(observation)
    targets = belief_targets(privileged)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask])


def hidden_hand_recall(
    logits: torch.Tensor,
    observation: torch.Tensor,
    privileged: torch.Tensor,
) -> tuple[float, float, int]:
    """Top-k opponent-card recall and the count-only uniform baseline."""
    probabilities = logits.softmax(dim=-1)[..., OPPONENT]
    unknown = candidate_mask(observation, include_known=False)
    truth = privileged[..., :NUM_CARDS] > 0.5
    if logits.ndim == 2:
        probabilities = probabilities.unsqueeze(0)
        unknown = unknown.unsqueeze(0)
        truth = truth.unsqueeze(0)
    recall_sum = 0.0
    baseline_sum = 0.0
    positions = 0
    for scores, mask, actual in zip(probabilities, unknown, truth):
        candidates = mask.nonzero(as_tuple=False).flatten()
        hidden_opponent = actual & mask
        needed = int(hidden_opponent.sum().item())
        # At deck=0 every remaining hidden card is necessarily in the hand;
        # that position does not test card-specific inference.
        if needed == 0 or len(candidates) == 0 or needed >= len(candidates):
            continue
        chosen_local = torch.topk(scores[candidates], min(needed, len(candidates))).indices
        chosen = candidates[chosen_local]
        recall_sum += float(hidden_opponent[chosen].sum().item()) / needed
        baseline_sum += needed / len(candidates)
        positions += 1
    return recall_sum, baseline_sum, positions
