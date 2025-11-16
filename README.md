# Tic-Tac-Toe Q-Learning AI
## Project Overview

This project demonstrates reinforcement learning by training an AI to play Tic-Tac-Toe
through trial and error, without being explicitly programmed with strategies.

## How to Run
1. **Train the AI**
```bash
python train_agent.py
```
Choose training intensity (1000-10000 games)

2. **Play Against AI**
```bash
python play_game.py
```

## How Q-Learning Works

### The Q-Table

The AI maintains a table of Q-values for each state-action pair:
- **State**: Current board configuration
- **Action**: Where to place the mark
- **Q-Value**: How good that action is in that state

### Learning Process

1. **Exploration**: Try random moves to discover strategies
2. **Exploitation**: Use learned knowledge to make good moves
3. **Reward**: Win = +10, Lose = -10, Tie = +5
4. **Update**: Adjust Q-values based on outcomes

### My Training Results

- Games trained: 10,000
- Final win rate: 85%
- States learned: 1,533

## What I Observed

I observed that the AI was able to beat 1/3 games that I played with it. It did respond by blocking me when I was able to get two x's in a row. Overall, it did make the correct moves and attempts but I was still able to win.

## Reinforcement Learning in Real Life

1. **Game AI**: Chess.com is a good example of a game that you can play against an AI that manually adjust difficulty based on the player's performance.
2. **Robotics**: Self-driving cars use AI to change tasks based off of movement around them, the fastest route, evading traffic jams, people crossing streets and parking lots.
3. **Recommendation Systems**: Spotify DJ is an excellent example that demonstrates learning patterns from users by their most frequently listened to albums and songs and making reccomendations of similar genres, albums and songs that others listened to with the same listening patterns.

## Challenges and Solutions
Initially, I wasn't sure how to train the model but I was able to access the options in the terminal and run the largest training option. It was a little bit difficult to understand the output of the training data to understand it in simplest terms so I responded to that challenge by doing more research on different states and AVG q-value.

## What I Learned
I learned that reinforcement learning is based on trial-and-error rather than being told exactly what to do, and the model improves by playing many rounds and updating its memory of what worked. I also learned that the Q-table stores information about which actions are good or bad depending on the current board state, and those values change as the agent gains more experience. I now understand that rewards guide the learning process, and the agent will favor actions that previously led to higher rewards or wins. Overall, reinforcement learning takes time, repetition, and evaluation instead of perfect instructions right away.