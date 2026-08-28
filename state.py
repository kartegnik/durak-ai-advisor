import torch
from durakgame import (Card, Suit, CardValue, Move, AttackingMove,
                       DefensiveMove, ForfeitingMove, EndingMove, OpeningMove,
                       FinishingMove, TransferMove, GameState)

# --- Card index mapping ---
# 4 suits × 9 values = 36 cards, index 0..35
SUIT_ORDER = [Suit.Heart, Suit.Cross, Suit.Spade, Suit.Diamond]
VALUE_ORDER = [CardValue.Six, CardValue.Seven, CardValue.Eight, CardValue.Nine,
               CardValue.Ten, CardValue.Jack, CardValue.Queen, CardValue.King, CardValue.Ace]

CARD_TO_INDEX = {}
INDEX_TO_CARD = {}
for si, suit in enumerate(SUIT_ORDER):
    for vi, value in enumerate(VALUE_ORDER):
        idx = si * 9 + vi
        card = Card(value, suit)
        CARD_TO_INDEX[card] = idx
        INDEX_TO_CARD[idx] = card

NUM_CARDS = 36
STATE_DIM = NUM_CARDS * 4 + 4 + 1 + 1  # 150
MOVE_TYPE_DIM = 7
OPTION_DIM = MOVE_TYPE_DIM + NUM_CARDS  # 43
COMBINED_DIM = STATE_DIM + OPTION_DIM  # 193


def card_to_onehot(card: Card) -> list[float]:
    vec = [0.0] * NUM_CARDS
    vec[CARD_TO_INDEX[card]] = 1.0
    return vec


def cards_to_multi_hot(cards) -> list[float]:
    vec = [0.0] * NUM_CARDS
    for card in cards:
        vec[CARD_TO_INDEX[card]] = 1.0
    return vec


def encode_state(hand: list[Card], state: GameState) -> list[float]:
    """Encode game state from player's perspective into a fixed-size vector."""
    vec = []
    # My hand (36)
    vec.extend(cards_to_multi_hot(hand))
    # Table attack (36)
    vec.extend(cards_to_multi_hot(state.table.attack))
    # Table defense (36)
    vec.extend(cards_to_multi_hot(state.table.defense))
    # Discard pile (36)
    vec.extend(cards_to_multi_hot(state.discardPile))
    # Trump suit one-hot (4)
    trump_vec = [0.0] * 4
    trump_vec[state.trumpCard.suit.value] = 1.0
    vec.extend(trump_vec)
    # Deck count normalized (1)
    vec.append(state.deckCC / 36.0)
    # Table forfeited (1)
    vec.append(1.0 if state.table.isForfeited else 0.0)
    return vec


def encode_option(move: Move) -> list[float]:
    """Encode a single move option into a fixed-size vector."""
    # Move type one-hot (7): Opening, Attack, Defense, Forfeit, End,
    # Finish, Transfer.
    type_vec = [0.0] * MOVE_TYPE_DIM
    card_vec = [0.0] * NUM_CARDS

    match move:
        case OpeningMove(card):
            type_vec[0] = 1.0
            card_vec[CARD_TO_INDEX[card]] = 1.0
        case AttackingMove(cards):
            type_vec[1] = 1.0
            for card in cards:
                card_vec[CARD_TO_INDEX[card]] = 1.0
        case DefensiveMove(card):
            type_vec[2] = 1.0
            card_vec[CARD_TO_INDEX[card]] = 1.0
        case ForfeitingMove():
            type_vec[3] = 1.0
        case EndingMove():
            type_vec[4] = 1.0
        case FinishingMove(cards):
            type_vec[5] = 1.0
            for card in cards:
                card_vec[CARD_TO_INDEX[card]] = 1.0
        case TransferMove(card):
            type_vec[6] = 1.0
            card_vec[CARD_TO_INDEX[card]] = 1.0
        case _:
            raise ValueError(f"Unsupported move type: {type(move).__name__}")

    return type_vec + card_vec


def encode_state_tensor(hand: list[Card], state: GameState) -> torch.Tensor:
    return torch.tensor(encode_state(hand, state), dtype=torch.float32)


def encode_options_tensor(options: list[Move]) -> torch.Tensor:
    return torch.tensor([encode_option(m) for m in options], dtype=torch.float32)


def encode_combined(hand: list[Card], state: GameState, options: list[Move]) -> torch.Tensor:
    """Returns tensor of shape [N, STATE_DIM + OPTION_DIM] — one row per option."""
    state_vec = encode_state_tensor(hand, state)  # [STATE_DIM]
    opts_vec = encode_options_tensor(options)  # [N, OPTION_DIM]
    state_expanded = state_vec.unsqueeze(0).expand(len(options), -1)  # [N, STATE_DIM]
    return torch.cat([state_expanded, opts_vec], dim=1)  # [N, COMBINED_DIM]
