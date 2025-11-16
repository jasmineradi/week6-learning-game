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
        Choose an action using the epsilon-greedy strategy.

        When training, the agent sometimes explores (tries a random move)
        and sometimes exploits (uses the best move it has learned so far).
        """
        if not available_actions:
            # No legal moves left (board is full)
            return None

        # --- Exploration step: try a random move some of the time ---
        # If we are in training mode AND a random number is less than
        # the exploration_rate, we ignore the Q-table and just pick
        # a random available action. This helps the agent discover
        # new state-action pairs.
        if self.training and random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # --- Exploitation step: choose the best-known move from the Q-table ---
        best_action = None
        best_value = float('-inf')
        # Look at each possible action and check its Q-value
        for action in available_actions:
            q_value = self.get_q_value(state, action)
            # Keep track of the action with the highest Q-value
            if q_value > best_value:
                best_value = q_value
                best_action = action

        # If for some reason all Q-values are equal or uninitialized,
        # fall back to a random action so the agent still makes a move.
        if best_action is None:
            best_action = random.choice(available_actions)

        return best_action

    def update_q_value(self, state, action, reward, next_state, next_actions):
        """
        Update the Q-value for a given state-action pair.

        This is the core of Q-learning:
        new_Q = old_Q + learning_rate * (reward + discount * max_next_Q - old_Q)
        """

        # Build a unique key for this (state, action) pair so it can be
        # stored and retrieved from the Q-table dictionary.
        key = f"{state}_{action}"

        # Current Q-value for this state-action (defaults to 0.0 if not seen before)
        current_q = self.get_q_value(state, action)

        # Look ahead to the next state: what is the best possible future value?
        if next_actions:
            # If there are legal next actions, get the maximum Q-value among them.
            max_next_q = max([self.get_q_value(next_state, a)
                              for a in next_actions])
        else:
            # If there are no next actions (game over), there is no future reward.
            max_next_q = 0

        # Apply the Q-learning update rule:
        #   - reward: immediate score from this action (win/lose/tie)
        #   - discount_factor * max_next_q: best future value, discounted
        #   - learning_rate controls how fast we adjust the old value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Store the updated Q-value back into the Q-table
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