# GPT-01 | Reproduce
# Actor-Critic Reinforcement Learning (RL) Implementation

This repository contains an implementation of an Actor-Critic reinforcement learning algorithm using PyTorch. The code provides a flexible and configurable framework for training neural networks that can serve as policy (actor) and value (critic) estimators for discrete action spaces.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements:
- **PolicyNetwork (Actor)**: Outputs logits over discrete actions.
- **ValueNetwork (Critic)**: Estimates the value of a given state.
- **RewardModel**: Defines a simple custom reward function.
- **ActorCriticAgent**: Coordinates the training process using a single optimizer for both actor and critic networks.
- **Training Loop**: Simulates a training process over random inputs to update the networks.

## Features
- Modular design with separate policy and value networks.
- Configurable training parameters through JSON configuration files.
- Automatic model checkpointing during training.
- Logging of training progress and errors.
- GPU/CPU support with automatic device detection.
- Deterministic training through seed setting.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/actor-critic-rl.git
cd actor-critic-rl

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Usage
### Training the Agent
```bash
python main.py
```
By default, the script will generate a `config.json` if it doesn't exist and start training using the default parameters.

### Custom Configuration
To modify training parameters, edit the `config.json` file or create one manually. Example:
```json
{
    "input_dim": 10,
    "output_dim": 2,
    "hidden_dim": 256,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "num_epochs": 100,
    "seed": 42
}
```

### Model Checkpoints
Model checkpoints are saved periodically during training in a `checkpoints_YYYYMMDD_HHMMSS` directory.

## Training Process
1. **Initialization**: Neural networks (policy and value) are initialized with random weights.
2. **Batch Sampling**: A batch of random states is generated at each epoch.
3. **Forward Pass**:
   - The policy network predicts action logits.
   - The value network estimates the state value.
4. **Reward Calculation**: Rewards are computed using the `RewardModel`.
5. **Loss Calculation**:
   - **Policy Loss** (Actor): Encourages selecting high-reward actions.
   - **Value Loss** (Critic): Trains the value network to predict actual rewards.
6. **Optimization**: The networks are updated using Adam optimizer.

## Configuration
The configuration is handled by the `Config` class, allowing flexible parameter tuning. The configuration can be saved and loaded from JSON files.

## File Structure
```
.
├── app.py                  # Main entry point for training
├── config.json              # Training configuration (auto-generated)
└── checkpoints/             # Model checkpoints
```

## Logging
Training logs are saved to `actor_critic_training.log` and streamed to the console. This includes information on:
- Epoch progress
- Loss values
- Evaluation results

## Contributing
Feel free to submit issues or pull requests if you encounter bugs or have suggestions for improvements.

## License
This project is licensed under the MIT License.
