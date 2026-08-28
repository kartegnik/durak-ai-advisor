import random
import torch
from durakgame import Player, Move, GameState
from model import DurakNet
from state import encode_combined


class NNPlayer(Player):
    """Neural network player for Durak."""

    def __init__(self, model: DurakNet | None = None, epsilon: float = 0.0):
        super().__init__()
        self.model = model or DurakNet()
        self.model.eval()
        self.epsilon = epsilon

    def nextMove(self, state: GameState, options: list[Move]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(options) - 1)

        combined = encode_combined(self.hand, state, options)
        return self.model.select_option(combined)

    def set_epsilon(self, eps: float):
        self.epsilon = eps

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        return self.model.load_compatible_state_dict(state_dict)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
