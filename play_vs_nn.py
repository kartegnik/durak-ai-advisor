#!/usr/bin/env python3
"""Play Durak: You vs Trained NNPlayer"""
from pathlib import Path

import torch
from durakgame import GameResult, standardDeck, GameConfiguration
from durakgame.game import Game
from model import DurakNet
from player import NNPlayer

PROJECT = Path(__file__).resolve().parent

SUIT_SYMBOLS = {'Heart': '♥', 'Cross': '♣', 'Spade': '♠', 'Diamond': '♦'}
# CardValue: Six=1, Seven=2, ..., Ace=9 (auto() starts from 1)
VALUE_NAMES = {1: '6', 2: '7', 3: '8', 4: '9', 5: '10', 6: 'J', 7: 'Q', 8: 'K', 9: 'A'}

def card_str(card):
    return f"{VALUE_NAMES[card.value.value]}{SUIT_SYMBOLS[card.suit.name]}"

def hand_str(hand):
    return ' '.join(f"[{i}]{card_str(c)}" for i, c in enumerate(hand))

def table_str(table):
    parts = []
    for i, (a, d) in enumerate(zip(table.attack, table.defense + [None]*(len(table.attack)-len(table.defense)))):
        if d:
            parts.append(f"{card_str(a)}-{card_str(d)}")
        else:
            parts.append(f"{card_str(a)}?")
    return ' '.join(parts)

class HumanPlayer:
    def __init__(self):
        self.hand = []
        self.game = None  # set after Game creation

    def nextMove(self, state, options):
        nn_cards = len(self.game.player1.hand) if self.game else "?"
        print(f"\n  Козырь: {card_str(state.trumpCard)}  Колода: {state.deckCC}  Карт у нейросети: {nn_cards}")
        print(f"  Стол: {table_str(state.table)}")
        print(f"  Отбой: {len(state.discardPile)} карт")
        print(f"  Твоя рука: {hand_str(self.hand)}")
        print(f"  Варианты:")
        for i, opt in enumerate(options):
            print(f"    [{i}] {opt}")
        while True:
            try:
                choice = int(input("  Твой ход: "))
                if 0 <= choice < len(options):
                    return choice
            except (ValueError, EOFError):
                pass
            print("  Неверный ввод, попробуй снова")

def main():
    # Load trained model
    model = DurakNet()
    try:
        ckpt = torch.load(PROJECT / "durak_model_v2.pt", weights_only=True)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            migrated = model.load_compatible_state_dict(ckpt['model'])
        else:
            migrated = model.load_compatible_state_dict(ckpt)
        print(f"Модель загружена (эпизод {ckpt.get('episode', '?')})")
        if migrated:
            print("Старые веса адаптированы для переводного хода; рекомендуется переобучение")
    except FileNotFoundError:
        print("Модель не найдена, играю случайно")

    nn_player = NNPlayer(model)
    human = HumanPlayer()

    # NN is player1, human is player2
    print("\n=== ДУРАК: Ты vs Нейросеть ===")
    print("Ты — player2 (защищаешься второй)")
    deck = standardDeck()
    import random
    random.shuffle(deck)

    g = Game(deck, nn_player, human, config=GameConfiguration(allowTransfer=True))
    human.game = g
    print(f"Козырь: {card_str(g.trump)}")

    g.kickoff()

    print(f"\n=== ИГРА ОКОНЧЕНА ===")
    print(f"Результат для player2 (ты): {g.result.name}")
    if g.result == GameResult.Win:
        print("Ты проиграл! (player1 победил)")
    elif g.result == GameResult.Loss:
        print("Ты победил!")
    else:
        print("Ничья / не доиграли")

if __name__ == "__main__":
    main()
