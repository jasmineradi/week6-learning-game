# play_game.py

from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent


def play_against_ai():
    """Play against the trained AI"""
    game = TicTacToe()
    agent = QLearningAgent(player_id=-1)  # AI plays as O

    # Load trained model
    if not agent.load_model("trained_model.json"):
        print("No trained model found! Run train_agent.py first!")
        return

    agent.set_training(False)  # Playing mode

    print("\n" + "=" * 60)
    print("🎮 PLAY AGAINST THE AI")
    print("=" * 60)
    print("You are X, AI is O")
    print("Enter moves as: row,col (e.g., 1,2)")

    # Randomly decide who goes first
    import random
    human_turn = random.choice([True, False])

    if not human_turn:
        print("\nAI goes first!")
    else:
        print("\nYou go first!")

    game.reset()

    while True:
        game.display()

        if human_turn:
            # Human's turn
            available = game.get_available_actions()
            print(f"\nAvailable moves: {available}")

            while True:
                try:
                    move = input("Your move (row,col): ")
                    row, col = map(int, move.split(','))
                    if (row, col) in available:
                        game.make_move((row, col), 1)
                        break
                    else:
                        print("Invalid move! Try again.")
                except:
                    print("Enter as: row,col (e.g., 0,1)")
        else:
            # AI's turn
            print("\nAI is thinking...")
            state = game.get_state()
            available = game.get_available_actions()

            # AI uses negative board values
            ai_state = str([-x for x in game.board.flatten()])
            action = agent.choose_action(ai_state, available)

            if action:
                game.make_move(action, -1)
                print(f"AI moves to: {action}")

        # Check for winner
        winner = game.check_winner()
        if winner is not None:
            game.display()

            if winner == 1:
                print("\n🎉 YOU WIN! Great job!")
            elif winner == -1:
                print("\n🤖 AI WINS! It learned well!")
            else:
                print("\n🤝 IT'S A TIE!")

            break

        human_turn = not human_turn


def main():
    """Main game menu"""
    while True:
        print("\n" + "=" * 60)
        print("🎮 TIC-TAC-TOE WITH Q-LEARNING AI")
        print("=" * 60)
        print("1. Play against trained AI")
        print("2. Watch AI vs AI")
        print("3. Learn about Q-Learning")
        print("4. Exit")

        choice = input("\nChoice (1-4): ")

        if choice == '1':
            play_against_ai()

        elif choice == '2':
            print("\n🤖 AI VS AI")
            print("Feature coming soon!")

        elif choice == '3':
            print("\n📚 ABOUT Q-LEARNING")
            print("-" * 40)
            print("Q-Learning is a reinforcement learning technique where:")
            print("• Q stands for 'Quality' of an action")
            print("• The AI learns Q-values for each state-action pair")
            print("• Higher Q-values mean better actions")
            print("\nThe Learning Process:")
            print("1. Try an action")
            print("2. Get a reward (win=+10, lose=-10, tie=+5)")
            print("3. Update Q-value based on reward")
            print("4. Repeat thousands of times")
            print("\nOver time, the AI learns which moves lead to wins!")

        elif choice == '4':
            print("\nThanks for playing! 👋")
            break


if __name__ == "__main__":
    main()