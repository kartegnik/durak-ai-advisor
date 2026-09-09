"""Losses for search-policy distillation and terminal outcome prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from durak_v4.belief import belief_cross_entropy


def alphazero_loss(model, trajectories, outcomes, belief_coefficient: float = 0.1):
    policy_losses = []
    value_losses = []
    belief_losses = []
    entropies = []
    for steps, outcome in zip(trajectories, outcomes):
        hidden = model.initial_hidden()
        belief_hidden = model.initial_belief_hidden()
        for step in steps:
            belief_logits, belief_hidden = model.belief_step(
                step.observation, belief_hidden,
            )
            logits, value, hidden = model.forward_step(
                step.observation, step.options, hidden, step.privileged,
                belief_hidden,
            )
            log_probabilities = logits.log_softmax(dim=0)
            probabilities = log_probabilities.exp()
            policy_losses.append(-(step.policy_target * log_probabilities).sum())
            value_losses.append(F.mse_loss(value, value.new_tensor(outcome)))
            belief_losses.append(belief_cross_entropy(
                belief_logits, step.observation, step.privileged,
            ))
            entropies.append(-(probabilities * log_probabilities).sum())

    if not policy_losses:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"policy": 0.0, "value": 0.0, "belief": 0.0, "entropy": 0.0}
    policy = torch.stack(policy_losses).mean()
    value = torch.stack(value_losses).mean()
    belief = torch.stack(belief_losses).mean()
    entropy = torch.stack(entropies).mean()
    total = policy + value + belief_coefficient * belief
    return total, {
        "policy": float(policy.detach()),
        "value": float(value.detach()),
        "belief": float(belief.detach()),
        "entropy": float(entropy.detach()),
    }
