# 🐦 Flappy Bird AI using Deep Q-Network (DQN)

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to play Flappy Bird through Reinforcement Learning.

Instead of being explicitly programmed with rules, the agent learns by interacting with the environment, collecting experiences, and improving its decision-making over time through trial and error.

The project is built using PyTorch and Gymnasium and demonstrates key Reinforcement Learning concepts such as:

* Deep Q-Learning
* Experience Replay
* Target Networks
* Epsilon-Greedy Exploration
* Bellman Equation
* Neural Network Function Approximation

---

## Demo

The agent observes the game state and decides between:

* **0 → Do Nothing**
* **1 → Flap**

Its goal is to maximize cumulative reward by surviving as long as possible and successfully navigating through pipes.

---

## How Deep Q-Learning Works

The agent repeatedly follows the cycle:

```text
Observe State
      ↓
Choose Action
      ↓
Execute Action
      ↓
Receive Reward
      ↓
Observe Next State
      ↓
Store Experience
      ↓
Learn From Experience
      ↓
Repeat
```

A neural network is trained to estimate:

Q(State, Action)

which represents the expected future reward of taking a particular action in a given state.

The learning target is computed using the Bellman Equation:

Q(s,a) = r + γ max Q(s',a')

where:

* s = current state
* a = current action
* r = immediate reward
* γ = discount factor
* s' = next state

---

## Features

### Deep Q-Network (DQN)

A neural network is used to approximate Q-values for each possible action.

Current architecture:

```python
nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, action_dim)
)
```

---

### Experience Replay

The agent stores experiences in a replay buffer:

```python
(state,
 action,
 next_state,
 reward,
 done)
```

Random mini-batches are sampled during training to reduce correlation between consecutive experiences and improve stability.

---

### Target Network

A separate target network is maintained to provide stable learning targets.

The target network is periodically synchronized with the policy network:

```python
target_dqn.load_state_dict(
    policy_dqn.state_dict()
)
```

This prevents the learning target from changing too rapidly.

---

### Epsilon-Greedy Exploration

The agent balances:

* Exploration (random actions)
* Exploitation (best known action)

During early training:

```text
epsilon ≈ 1.0
```

The agent mostly explores.

As training progresses:

```text
epsilon → 0.05
```

The agent increasingly relies on learned knowledge.

---

## Project Structure

```text
FlappyBird/
│
├── agent.py
├── dqn.py
├── experience_replay.py
├── parameters.yaml
│
├── runs/
│   ├── flappybirdv0.pt
│   └── flappybirdv0.log
│
└── README.md
```

### Files

#### agent.py

Main training and inference pipeline.

Responsibilities:

* Environment interaction
* Experience collection
* Training loop
* Model saving/loading
* Target network synchronization

#### dqn.py

Defines the Deep Q-Network architecture.

#### experience_replay.py

Implements the replay memory buffer used during training.

#### parameters.yaml

Stores all configurable hyperparameters.

---

## Hyperparameters

Example configuration:

```yaml
flappybirdv0:
  epsilon_init: 1
  epsilon_min: 0.05
  epsilon_decay: 0.9995
  replay_memory_size: 10000
  mini_batch_size: 32
  network_sync_rate: 10
  alpha: 0.001
  gamma: 0.99
  reward_threshold: 1000
```

### Parameter Explanation

| Parameter          | Description                         |
| ------------------ | ----------------------------------- |
| alpha              | Learning rate                       |
| gamma              | Discount factor                     |
| epsilon_init       | Initial exploration probability     |
| epsilon_min        | Minimum exploration probability     |
| epsilon_decay      | Exploration decay rate              |
| replay_memory_size | Maximum replay buffer size          |
| mini_batch_size    | Number of samples per training step |
| network_sync_rate  | Frequency of target network updates |
| reward_threshold   | Episode reward cap                  |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/flappy-bird-dqn.git
cd flappy-bird-dqn
```

### Create Virtual Environment

```bash
python -m venv ml-env
```

### Activate Environment

#### Windows

```bash
ml-env\Scripts\activate
```

#### Linux / Mac

```bash
source ml-env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train the agent:

```bash
python agent.py flappybirdv0 --train
```

During training:

* Experiences are collected
* Replay memory is populated
* Mini-batches are sampled
* Neural network weights are updated
* Best-performing models are saved automatically

Training logs are stored inside:

```text
runs/flappybirdv0.log
```

Best model checkpoints are stored inside:

```text
runs/flappybirdv0.pt
```

---

## Running a Trained Agent

After training:

```bash
python agent.py flappybirdv0
```

The saved model is loaded and used for inference.

No exploration is performed during testing.

The agent always chooses the action with the highest predicted Q-value.

---

## Learning Outcomes

This project helped me understand:

* Reinforcement Learning fundamentals
* Deep Q-Learning
* Bellman Equation
* Neural Networks in Reinforcement Learning
* Experience Replay
* Target Networks
* Exploration vs Exploitation
* PyTorch Training Workflows
* Environment Interaction using Gymnasium

---

## Future Improvements

Potential extensions:

* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* TensorBoard Integration
* Reward Shaping Experiments
* Hyperparameter Optimization
* Multi-Layer Network Architectures
* Model Checkpointing
* Performance Visualization

---

## Acknowledgements

* PyTorch
* Gymnasium
* flappy-bird-gymnasium

---

## License

This project is intended for educational and learning purposes.
