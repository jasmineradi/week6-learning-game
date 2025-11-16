# train_agent.py

from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
import random


class Trainer:
    """Train the Q-learning agent to play Tic-Tac-Toe"""

    def __init__(self):
        self.game = TicTacToe()
        self.agent = QLearningAgent(player_id=1)  # Agent is X
        self.wins = {'agent': 0, 'random': 0, 'tie': 0}

    def play_training_game(self):
        """Play one training game"""
        self.game.reset()
        states_actions_rewards = []

        while True:
            # Agent's turn (X)
            state = self.game.get_state()
            available_actions = self.game.get_available_actions()

            if not available_actions:
                break

            action = self.agent.choose_action(state, available_actions)
            self.game.make_move(action, 1)

            # Store for learning
            states_actions_rewards.append((state, action, 0))  # Reward comes later

            # Check winner
            winner = self.game.check_winner()
            if winner is not None:
                self.process_game_end(winner, states_actions_rewards)
                break

            # Random opponent's turn (O)
            available_actions = self.game.get_available_actions()
            if available_actions:
                opponent_action = random.choice(available_actions)
                self.game.make_move(opponent_action, -1)

                # Check winner again
                winner = self.game.check_winner()
                if winner is not None:
                    self.process_game_end(winner, states_actions_rewards)
                    break

    def process_game_end(self, winner, states_actions_rewards):
        """Assign rewards based on game outcome"""
        if winner == 1:  # Agent won
            reward = 10
            self.wins['agent'] += 1
        elif winner == -1:  # Opponent won
            reward = -10
            self.wins['random'] += 1
        else:  # Tie
            reward = 5
            self.wins['tie'] += 1

        # Update Q-values for all moves (backward)
        states_actions_rewards.reverse()

        for i, (state, action, _) in enumerate(states_actions_rewards):
            if i == 0:
                # Last move gets full reward
                self.agent.update_q_value(state, action, reward, "", [])
            else:
                # Earlier moves get discounted reward
                next_state = states_actions_rewards[i - 1][0]
                next_actions = self.game.get_available_actions()
                self.agent.update_q_value(
                    state,
                    action,
                    reward * (0.9 ** i),
                    next_state,
                    next_actions
                )

    def train(self, episodes=1000):
        """Train the agent"""
        print("🎮 TRAINING TIC-TAC-TOE AI")
        print("=" * 50)

        checkpoints = [100, 500, 1000, 5000, 10000]

        for episode in range(1, episodes + 1):
            self.play_training_game()

            if episode in checkpoints or episode == episodes:
                win_rate = (self.wins['agent'] / episode) * 100
                print(f"\nEpisode {episode}:")
                print(f"  Wins: {self.wins['agent']} ({win_rate:.1f}%)")
                print(f"  Losses: {self.wins['random']}")
                print(f"  Ties: {self.wins['tie']}")

                stats = self.agent.get_stats()
                print(f"  States learned: {stats['states_learned']}")
                print(f"  Avg Q-value: {stats['avg_q_value']:.3f}")

                # Reduce exploration over time
                if episode == 500:
                    self.agent.exploration_rate = 0.2
                elif episode == 1000:
                    self.agent.exploration_rate = 0.1
                elif episode == 5000:
                    self.agent.exploration_rate = 0.05

        # Save trained model
        self.agent.save_model("trained_model.json")

        return self.agent


def main():
    """Main training program"""
    print("\n" + "=" * 60)
    print("🤖 REINFORCEMENT LEARNING DEMONSTRATION")
    print("Training an AI to play Tic-Tac-Toe")
    print("=" * 60)

    trainer = Trainer()

    # Training options
    print("\nTraining Options:")
    print("1. Quick training (1,000 games)")
    print("2. Standard training (5,000 games)")
    print("3. Intensive training (10,000 games)")

    choice = input("\nChoice (1-3): ")

    episodes = {
        '1': 1000,
        '2': 5000,
        '3': 10000
    }.get(choice, 1000)

    print(f"\nTraining with {episodes} games...")
    print("Watch the AI improve!\n")

    trained_agent = trainer.train(episodes)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nThe AI has learned to play Tic-Tac-Toe!")
    print("Run play_game.py to play against it!")

    # Show what the AI learned
    print("\n📚 WHAT THE AI LEARNED:")
    print("-" * 40)
    print("1. Opening moves are important")
    print("2. Center and corners are valuable")
    print("3. Block opponent's winning moves")
    print("4. Create winning opportunities")
    print("\nThis happened through trial and error,")
    print("not by programming rules!")


if __name__ == "__main__":
    main()