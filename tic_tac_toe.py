# tic_tac_toe.py

import numpy as np
import random


class TicTacToe:
    """
    Tic-Tac-Toe game environment
    Perfect for demonstrating reinforcement learning!
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game board"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player 1 starts
        return self.get_state()

    def get_state(self):
        """Convert board to a string state for Q-learning"""
        return str(self.board.flatten())

    def get_available_actions(self):
        """Return list of empty positions"""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions

    def make_move(self, action, player):
        """Make a move on the board"""
        row, col = action
        if self.board[row, col] == 0:
            self.board[row, col] = player
            return True
        return False

    def check_winner(self):
        """Check if someone won"""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]

        # Check columns
        for col in range(3):
            if self.board[0, col] == self.board[1, col] == self.board[2, col] != 0:
                return self.board[0, col]

        # Check diagonals
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return self.board[0, 0]
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return self.board[0, 2]

        # Check for tie
        if not self.get_available_actions():
            return 0  # Tie

        return None  # Game continues

    def display(self):
        """Show the board nicely"""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("\n   0   1   2")
        print("  -----------")
        for i in range(3):
            print(f"{i}| ", end="")
            for j in range(3):
                print(f"{symbols[self.board[i, j]]} | ", end="")
            print("\n  -----------")