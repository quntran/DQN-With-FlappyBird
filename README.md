# DQN-With-FlappyBird

A Deep Q-Learning (DQN) implementation for playing Flappy Bird using PyTorch. This project implements both standard DQN and advanced variants including Double DQN and Dueling DQN architectures.

## Overview

This project trains an AI agent to play Flappy Bird using deep reinforcement learning. The agent learns to navigate through pipes by experiencing the game and improving its policy through trial and error.

### Features

- **Deep Q-Network (DQN)**: Neural network-based Q-learning
- **Double DQN**: Reduces overestimation of Q-values
- **Dueling DQN**: Separates value and advantage streams for better learning
- **Experience Replay**: Stores and samples past experiences to break correlation
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Configurable Hyperparameters**: Easy tuning via YAML configuration
- **Training Visualization**: Automatic plotting of rewards and epsilon decay
- **Model Persistence**: Automatic saving of best-performing models

## Project Structure

```
.
├── agent.py                 # Main DQN agent implementation
├── dqn.py                   # Deep Q-Network architecture
├── experience_replay.py     # Experience replay memory
├── hyperparameters.yml      # Hyperparameter configurations
├── requirements.txt         # Python dependencies
└── runs/                    # Training outputs (logs, models, graphs)
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional, but recommended for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/quntran/DQN-With-FlappyBird.git
cd DQN-With-FlappyBird
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the agent:

```python
from agent import Agent

# Create agent with hyperparameter set
dql = Agent("flappybird1")

# Start training (no rendering for faster training)
dql.run(is_training=True, render=False)
```

Or modify the `__main__` section in `agent.py` and run:
```bash
python agent.py
```

### Testing/Inference

To watch a trained agent play:

```python
from agent import Agent

# Load trained agent
dql = Agent("flappybird1")

# Run with trained model (with rendering)
dql.run(is_training=False, render=True)
```

## Configuration

Hyperparameters are defined in `hyperparameters.yml`. The project includes configurations for both CartPole and Flappy Bird environments.

### Flappy Bird Configuration (`flappybird1`)

```yaml
env_id: FlappyBird-v0
replay_memory_size: 100000    # Size of experience replay buffer
mini_batch_size: 32           # Batch size for training
epsilon_init: 1               # Initial exploration rate
epsilon_decay: 0.99995        # Exploration decay rate
epsilon_min: 0.01             # Minimum exploration rate
network_sync_rate: 10         # Steps between target network updates
learning_rate_a: 0.0001       # Learning rate (alpha)
discount_factor_g: 0.99       # Discount factor (gamma)
stop_on_reward: 100000        # Maximum episode reward
fc1_nodes: 512                # First hidden layer size
fc2_nodes: 256                # Second hidden layer size
enable_double_dqn: True       # Enable Double DQN
enable_dueling_dqn: True      # Enable Dueling DQN
```

## Architecture

### DQN Network

The neural network consists of:
- **Input Layer**: State dimensions (varies by environment)
- **Hidden Layers**: Two fully connected layers (configurable sizes)
- **Output Layer**: Q-values for each action

**Dueling DQN variant** splits the network into:
- **Value Stream**: Estimates state value V(s)
- **Advantage Stream**: Estimates action advantages A(s,a)
- **Combined Output**: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

### Training Algorithm

1. Initialize policy network and target network
2. For each episode:
   - Observe current state
   - Select action using epsilon-greedy policy
   - Execute action and observe reward and next state
   - Store experience in replay memory
   - Sample mini-batch from replay memory
   - Compute target Q-values using target network
   - Update policy network to minimize TD error
   - Periodically sync target network with policy network
3. Save model when achieving new best reward

### Double DQN

Standard DQN can overestimate Q-values. Double DQN addresses this by:
- Using the **policy network** to select the best action
- Using the **target network** to evaluate that action's value

## Training Output

During training, the agent generates:

- **Log file** (`runs/{hyperparameter_set}.log`): Training progress and best rewards
- **Model file** (`runs/{hyperparameter_set}.pt`): Saved neural network weights
- **Graph file** (`runs/{hyperparameter_set}.png`): Plots showing:
  - Mean rewards over episodes
  - Epsilon decay over time

## Performance

The agent uses:
- Experience replay to break temporal correlations
- Target network updates to stabilize training
- Epsilon decay for gradual shift from exploration to exploitation
- Best model checkpointing to preserve optimal performance

Training is designed to run indefinitely. Monitor the graphs and stop training when satisfied with performance.

## Environment

This project uses the [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) environment, which provides a Gymnasium-compatible interface for Flappy Bird.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Quang Tran

## Acknowledgments

- Based on the DQN algorithm from DeepMind's "Playing Atari with Deep Reinforcement Learning"
- Uses the Gymnasium framework for reinforcement learning environments
- Flappy Bird environment from flappy-bird-gymnasium package
