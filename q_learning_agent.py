# q_learning_agent.py

import random
import json
import os


class QLearningAgent:
    """
    An agent that learns to play Tic-Tac-Toe using Q-learning
    This demonstrates reinforcement learning from Chapter 21!
    """

    def __init__(self, player_id, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=0.3):
        """
        Initialize Q-learning agent

        learning_rate: How much new info overrides old
        discount_factor: Importance of future rewards
        exploration_rate: How often to try random moves
        """
        self.player_id = player_id
        self.q_table = {}  # Stores Q-values for state-action pairs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.training = True

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        key = f"{state}_{action}"
        if key not in self.q_table:
            self.q_table[key] = 0.0
        return self.q_table[key]

    def choose_action(self, state, available_actions):
        """
        Choose action using epsilon-greedy strategy
        Sometimes explore (random), sometimes exploit (best known)
        """
        if not available_actions:
            return None

        # Exploration: random action
        if self.training and random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Exploitation: best known action
        best_action = None
        best_value = float('-inf')

        for action in available_actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        # If all Q-values are the same, choose randomly
        if best_action is None:
            best_action = random.choice(available_actions)

        return best_action

    def update_q_value(self, state, action, reward, next_state, next_actions):
        """
        Update Q-value using the Q-learning formula
        This is where the learning happens!
        """
        key = f"{state}_{action}"

        # Get current Q-value
        current_q = self.get_q_value(state, action)

        # Get maximum Q-value for next state
        if next_actions:
            max_next_q = max([self.get_q_value(next_state, a)
                              for a in next_actions])
        else:
            max_next_q = 0

        # Q-learning formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[key] = new_q

    def set_training(self, training):
        """Switch between training and playing mode"""
        self.training = training
        if not training:
            self.exploration_rate = 0  # No exploration when playing

    def save_model(self, filepath):
        """Save Q-table to file"""
        with open(filepath, 'w') as f:
            json.dump(self.q_table, f, indent=2)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load Q-table from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.q_table = json.load(f)
            print(f"Model loaded from {filepath}")
            return True
        return False

    def get_stats(self):
        """Get learning statistics"""
        return {
            'states_learned': len(set(k.split('_')[0] for k in self.q_table)),
            'total_q_values': len(self.q_table),
            'avg_q_value': sum(self.q_table.values()) / len(self.q_table) if self.q_table else 0
        }