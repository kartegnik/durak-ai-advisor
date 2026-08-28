"""Recurrent PPO losses with terminal rewards and generalized advantages."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .agents import RolloutStep
from .model import RecurrentActorCritic


@dataclass
class PreparedTrajectory:
    steps: list[RolloutStep]
    advantages: torch.Tensor
    returns: torch.Tensor


def prepare_trajectory(
    steps: list[RolloutStep], terminal_reward: float,
    gamma: float = 0.997, gae_lambda: float = 0.95,
) -> PreparedTrajectory:
    if not steps:
        raise ValueError("empty trajectory")
    advantages = torch.zeros(len(steps), dtype=torch.float32)
    gae = 0.0
    next_value = 0.0
    for index in range(len(steps) - 1, -1, -1):
        terminal = index == len(steps) - 1
        reward = steps[index].reward + (terminal_reward if terminal else 0.0)
        continuation = 0.0 if terminal else 1.0
        delta = reward + gamma * next_value * continuation - steps[index].old_value
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[index] = gae
        next_value = steps[index].old_value
    values = torch.tensor([step.old_value for step in steps])
    return PreparedTrajectory(steps, advantages, advantages + values)


def normalize_advantages(trajectories: list[PreparedTrajectory]) -> None:
    combined = torch.cat([trajectory.advantages for trajectory in trajectories])
    mean = combined.mean()
    std = combined.std(unbiased=False).clamp_min(1e-6)
    for trajectory in trajectories:
        trajectory.advantages = (trajectory.advantages - mean) / std


def ppo_update(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    trajectories: list[PreparedTrajectory],
    epochs: int = 4,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.015,
    tactical_coefficient: float = 0.15,
    belief_coefficient: float = 0.0,
    max_grad_norm: float = 0.8,
) -> dict[str, float]:
    normalize_advantages(trajectories)
    totals = {
        "loss": 0.0, "policy": 0.0, "value": 0.0,
        "entropy": 0.0, "tactical": 0.0, "belief": 0.0,
    }
    for _ in range(epochs):
        new_log_probs = []
        new_values = []
        entropies = []
        old_log_probs = []
        advantages = []
        returns = []
        tactical_losses = []
        belief_losses = []
        # Advance every trajectory at the same recurrent timestep together.
        # Legal option counts remain variable, so their rows are concatenated
        # for one policy pass and split back into per-state distributions.
        hidden = torch.zeros(len(trajectories), model.hidden_size)
        belief_hidden = (
            torch.zeros(len(trajectories), model.hidden_size)
            if belief_coefficient > 0.0 and hasattr(model, "belief_step") else None
        )
        max_length = max(len(trajectory.steps) for trajectory in trajectories)
        for timestep in range(max_length):
            active = [
                index for index, trajectory in enumerate(trajectories)
                if timestep < len(trajectory.steps)
            ]
            if not active:
                continue
            active_tensor = torch.tensor(active, dtype=torch.long)
            steps = [trajectories[index].steps[timestep] for index in active]
            observations = torch.stack([step.observation for step in steps])
            privileged = torch.stack([step.privileged for step in steps])
            encoded = model.observation_encoder(observations)
            next_hidden = model.memory(encoded, hidden.index_select(0, active_tensor))
            hidden = hidden.index_copy(0, active_tensor, next_hidden)

            if belief_hidden is not None:
                from durak_v4.belief import belief_cross_entropy
                belief_logits, next_belief_hidden = model.belief_step(
                    observations, belief_hidden.index_select(0, active_tensor),
                )
                belief_hidden = belief_hidden.index_copy(
                    0, active_tensor, next_belief_hidden,
                )
                belief_losses.append(belief_cross_entropy(
                    belief_logits, observations, privileged,
                ))

            policy_hidden = model.policy_context(
                next_hidden,
                next_belief_hidden if belief_hidden is not None else None,
            )

            privileged_features = model.privileged_encoder(privileged)
            value_batch = model.value(
                torch.cat((next_hidden, privileged_features), dim=-1)
            ).squeeze(-1)
            lengths = [len(step.options) for step in steps]
            all_options = torch.cat([step.options for step in steps])
            option_features = model.option_encoder(all_options)
            contexts = torch.repeat_interleave(
                policy_hidden, torch.tensor(lengths), dim=0,
            )
            all_logits = model.policy(
                torch.cat((contexts, option_features), dim=-1)
            ).squeeze(-1)
            split_logits = all_logits.split(lengths)

            for local_index, (trajectory_index, step, logits) in enumerate(
                zip(active, steps, split_logits)
            ):
                distribution = Categorical(logits=logits)
                action = torch.tensor(step.action)
                new_log_probs.append(distribution.log_prob(action))
                new_values.append(value_batch[local_index])
                entropies.append(distribution.entropy())
                if step.tactical_rewards is not None:
                    rewards = step.tactical_rewards
                    if float(rewards.max() - rewards.min()) > 1e-6:
                        safe = rewards >= rewards.max() - 1e-6
                        tactical_losses.append(-torch.log(
                            distribution.probs[safe].sum().clamp_min(1e-8)
                        ))
                old_log_probs.append(step.old_log_prob)
                advantages.append(trajectories[trajectory_index].advantages[timestep])
                returns.append(trajectories[trajectory_index].returns[timestep])

        new_log_probs_t = torch.stack(new_log_probs)
        old_log_probs_t = torch.tensor(old_log_probs)
        advantages_t = torch.stack(advantages)
        returns_t = torch.stack(returns)
        values_t = torch.stack(new_values)
        entropy = torch.stack(entropies).mean()
        tactical_loss = (
            torch.stack(tactical_losses).mean()
            if tactical_losses else torch.tensor(0.0)
        )
        belief_loss = (
            torch.stack(belief_losses).mean()
            if belief_losses else torch.tensor(0.0)
        )

        ratio = (new_log_probs_t - old_log_probs_t).exp()
        unclipped = ratio * advantages_t
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_t
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = F.smooth_l1_loss(values_t, returns_t)
        loss = (
            policy_loss + value_coefficient * value_loss
            - entropy_coefficient * entropy
            + tactical_coefficient * tactical_loss
            + belief_coefficient * belief_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["policy"] += float(policy_loss.detach())
        totals["value"] += float(value_loss.detach())
        totals["entropy"] += float(entropy.detach())
        totals["tactical"] += float(tactical_loss.detach())
        totals["belief"] += float(belief_loss.detach())

    return {key: value / epochs for key, value in totals.items()}


def pretrain_belief(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    sequences: list[list[RolloutStep]],
    epochs: int = 3,
    sequence_batch_size: int = 16,
) -> float:
    """Teach recurrent memory to locate hidden cards before policy fine-tuning."""
    if not hasattr(model, "belief_step"):
        raise TypeError("model has no belief prediction head")
    from durak_v4.belief import belief_cross_entropy

    last_loss = 0.0
    for _ in range(epochs):
        random.shuffle(sequences)
        for start in range(0, len(sequences), sequence_batch_size):
            batch = sequences[start:start + sequence_batch_size]
            belief_hidden = torch.zeros(len(batch), model.hidden_size)
            losses = []
            for timestep in range(max(map(len, batch))):
                active = [index for index, sequence in enumerate(batch) if timestep < len(sequence)]
                active_tensor = torch.tensor(active, dtype=torch.long)
                steps = [batch[index][timestep] for index in active]
                observations = torch.stack([step.observation for step in steps])
                privileged = torch.stack([step.privileged for step in steps])
                logits, next_belief_hidden = model.belief_step(
                    observations, belief_hidden.index_select(0, active_tensor),
                )
                belief_hidden = belief_hidden.index_copy(
                    0, active_tensor, next_belief_hidden,
                )
                losses.append(belief_cross_entropy(
                    logits, observations, privileged,
                ))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            last_loss = float(loss.detach())
    return last_loss


def behavior_clone(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    sequences: list[list[tuple[torch.Tensor, torch.Tensor, int]]],
    epochs: int = 3,
    sequence_batch_size: int = 16,
) -> float:
    last_loss = 0.0
    for _ in range(epochs):
        random.shuffle(sequences)
        for start in range(0, len(sequences), sequence_batch_size):
            batch = sequences[start:start + sequence_batch_size]
            hidden = torch.zeros(len(batch), model.hidden_size)
            losses = []
            for timestep in range(max(map(len, batch))):
                active = [index for index, sequence in enumerate(batch) if timestep < len(sequence)]
                active_tensor = torch.tensor(active, dtype=torch.long)
                steps = [batch[index][timestep] for index in active]
                observations = torch.stack([step[0] for step in steps])
                encoded = model.observation_encoder(observations)
                next_hidden = model.memory(encoded, hidden.index_select(0, active_tensor))
                hidden = hidden.index_copy(0, active_tensor, next_hidden)
                lengths = [len(step[1]) for step in steps]
                options = torch.cat([step[1] for step in steps])
                option_features = model.option_encoder(options)
                contexts = torch.repeat_interleave(next_hidden, torch.tensor(lengths), dim=0)
                logits = model.policy(torch.cat((contexts, option_features), dim=-1)).squeeze(-1)
                for step, step_logits in zip(steps, logits.split(lengths)):
                    losses.append(F.cross_entropy(
                        step_logits.unsqueeze(0), torch.tensor([step[2]])
                    ))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            last_loss = float(loss.detach())
    return last_loss
