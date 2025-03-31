# Snake AI - Deep Learning Project
## Overview
This project utilizes deep learning techniques to train an artificial intelligence (AI) to play the classic Snake game. The AI learns by interacting with the game environment and making decisions based on the current state of the game. Using reinforcement learning, the agent aims to maximize its score by eating food while avoiding collisions with the walls or its own body.

## Features
* Reinforcement Learning (RL): The AI is trained using RL, where it learns from rewards (eating food) and penalties (colliding with walls or its body).
* Neural Network: A deep neural network (DNN) is used to approximate the optimal policy for playing the game.
* Game Environment: A simple Snake game environment is implemented for training and testing the AI.

Training & Evaluation: The model is trained over several episodes, and its performance is evaluated based on the score it achieves.

Requirements
* Python 3.x
* TensorFlow (for deep learning)
* NumPy
* Matplotlib (for visualizing the training progress)

Installation
1. Clone the repository
```bash
Copy
git clone https://github.com/yourusername/snake-ai.git
cd snake-ai
```
2. Install dependencies
```bash
Copy
pip install -r requirements.txt
```
## Usage
1. Running the Snake game
The game environment is implemented using Python, and the Snake AI interacts with it through actions and observations.

To run the Snake game with the trained model:
```bash
Copy
python run_game.py
```
This will start the game, and the AI will play the game using its trained policy.

2. Training the AI
To train the AI model, you can run the training script:

```bash
python train.py
```
The AI will start learning from scratch. During training, the agent will explore different actions, and the neural network will gradually improve its policy to play the game better. Training progress will be visualized using Matplotlib.

3. Visualizing Training Progress
If you want to visualize the training results, you can enable logging and plotting during training. The agent's performance and rewards per episode will be plotted as the training progresses.

```bash
python train.py --visualize
```
## Code Structure
game.py: The game environment implementation, including the logic for the Snake game and the rules.

agent.py: The AI agent, which includes the neural network and decision-making logic for the Snake AI.

train.py: The training script that implements the reinforcement learning algorithm and trains the agent.

run_game.py: The script to run the trained model and observe the AI playing the game.

utils.py: Helper functions for logging, visualization, and model saving.

Model Architecture
The AI agent uses a Deep Q-Network (DQN), which combines Q-learning with deep neural networks. The neural network takes the current state (the Snake game grid) as input and outputs Q-values for each possible action (up, down, left, right). The agent selects actions based on the highest Q-value.

Neural Network Architecture:
Input: Current state (grid representing Snake game)

Output: Q-values for each action (4 values for up, down, left, right)

Hidden layers: A few fully connected layers with ReLU activation functions

Training Algorithm
Exploration vs. Exploitation: The agent balances exploration (trying random actions) and exploitation (choosing the best-known action).

Replay Memory: The agent stores experiences in a replay buffer, which helps to break the correlation between consecutive observations and improve training stability.

Target Network: A target network is used to compute the Q-values for the next state to improve stability during training.

Loss Function: The loss function used is the mean squared error between predicted Q-values and target Q-values.

Results
After sufficient training, the AI is able to play the Snake game proficiently, navigating the snake towards food while avoiding collisions. Performance can be measured by tracking the highest score achieved during training.
