import torch
import torch.nn as nn
from state import COMBINED_DIM, STATE_DIM


class DurakNet(nn.Module):
    """Scores each (state, option) pair. Higher score = better move."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COMBINED_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, COMBINED_DIM] → scores: [N]"""
        return self.net(x).squeeze(-1)

    def select_option(self, combined: torch.Tensor) -> int:
        """Pick the option index with highest score.
        combined: [N, COMBINED_DIM]
        Returns: int index into options list.
        """
        with torch.no_grad():
            scores = self.forward(combined)
        return scores.argmax().item()

    def load_compatible_state_dict(self, state_dict: dict) -> bool:
        """Load current weights or migrate a pre-TransferMove checkpoint.

        Returns True when the legacy 192-input first layer was migrated to the
        current 193-input layout.  The new transfer-type weight is initialized
        to zero; retraining is still recommended.
        """
        first_layer_key = "net.0.weight"
        old_weight = state_dict.get(first_layer_key)
        new_weight = self.state_dict()[first_layer_key]

        if old_weight is None or old_weight.shape == new_weight.shape:
            self.load_state_dict(state_dict)
            return False

        if (old_weight.shape[0] != new_weight.shape[0]
                or old_weight.shape[1] + 1 != new_weight.shape[1]):
            self.load_state_dict(state_dict)
            return False

        migrated = dict(state_dict)
        migrated_weight = torch.zeros_like(new_weight)

        # State and six legacy move-type fields keep their positions.
        legacy_prefix = STATE_DIM + 6
        migrated_weight[:, :legacy_prefix] = old_weight[:, :legacy_prefix]
        # The 36 card fields move one column to make room for TransferMove.
        migrated_weight[:, legacy_prefix + 1:] = old_weight[:, legacy_prefix:]
        migrated[first_layer_key] = migrated_weight
        self.load_state_dict(migrated)
        return True
