import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import logging
import json
from pathlib import Path
from typing import List, Union, Optional, Dict
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('actor_critic_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

########################################
#               CONFIG
########################################

class Config:
    def __init__(self, **kwargs):
        self.input_dim = kwargs.get('input_dim', 10)
        self.output_dim = kwargs.get('output_dim', 2)
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.seed = kwargs.get('seed', 42)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # For this simple example, single-step means gamma doesn't matter.
        # But if you want multi-step returns, you can add gamma here.
        self.gamma = kwargs.get('gamma', 0.99)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

########################################
#               UTILS
########################################

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################################
#               MODELS
########################################

class PolicyNetwork(nn.Module):
    """
    Actor network outputting unnormalized logits for discrete actions.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self) -> None:
        for module in [self.layer1, self.layer2, self.output_layer]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        if isinstance(x, (list, np.ndarray)):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        logits = self.output_layer(x)
        return logits


class ValueNetwork(nn.Module):
    """
    Critic network outputting a single scalar value estimate (V(s)).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 1)
        self.init_weights()

    def init_weights(self) -> None:
        for module in [self.layer1, self.layer2, self.output_layer]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        if isinstance(x, (list, np.ndarray)):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x).squeeze(-1)

########################################
#           REWARD MODEL
########################################

class RewardModel:
    """
    Example reward function:
      - Base reward = 0.1
      - +1.0 if sum(state) > 1.5 and action == 1
      - +0.5 every 10th step
      - -0.5 if sum(state) < 0.5
    """
    def compute_reward(self, state: Union[torch.Tensor, np.ndarray],
                       action: int, step: int) -> float:
        if isinstance(state, torch.Tensor):
            state_sum = state.sum().item()
        elif isinstance(state, np.ndarray):
            state_sum = state.sum()
        else:
            raise ValueError("State must be either torch.Tensor or np.ndarray")

        reward = 0.1
        if state_sum > 1.5 and action == 1:
            reward += 1.0
        if step % 10 == 0:  # e.g. extra reward for every 10th step
            reward += 0.5
        if state_sum < 0.5:
            reward -= 0.5
        return float(reward)

########################################
#         ACTOR-CRITIC AGENT
########################################

class ActorCriticAgent:
    def __init__(self,
                 policy_network: PolicyNetwork,
                 value_network: ValueNetwork,
                 config: Config):
        self.policy_network = policy_network.to(config.device)
        self.value_network = value_network.to(config.device)
        self.config = config
        self.device = torch.device(config.device)

        # Single optimizer for both actor & critic (common in simple code);
        # you could also separate them if desired.
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=config.learning_rate
        )

        # Keep track of training metrics
        self.training_history: List[Dict[str, float]] = []

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_network": self.policy_network.state_dict(),
            "value_network": self.value_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "config": self.config.__dict__,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_history = checkpoint["training_history"]
        self.config = Config(**checkpoint["config"])

    def update(self, states: np.ndarray, step: int, reward_model: RewardModel) -> float:
        """
        Single-step Actor-Critic update (batch-mode).
        
        1) We get a batch of states.
        2) Sample an action for each state from the policy network.
        3) Compute reward for each (state, action).
        4) Advantage = reward - V(s).
        5) Actor (policy) loss = -log_prob(action) * advantage.
        6) Critic (value) loss = MSE( V(s), reward ).
        7) Optimize total_loss.
        """
        device = self.device

        # Convert states to tensor
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)  # [batch, input_dim]

        # === Actor Forward ===
        # Unnormalized logits for each action
        policy_logits = self.policy_network(states_tensor)
        dist = torch.distributions.Categorical(logits=policy_logits)

        # Sample actions
        actions = dist.sample()  # [batch]

        # Log-prob of each chosen action
        log_probs = dist.log_prob(actions)  # [batch]

        # === Critic Forward ===
        # Value estimates for each state
        values = self.value_network(states_tensor)  # [batch]

        # === Compute Rewards ===
        # We'll compute a reward for each (state, action).
        rewards_list = []
        for i in range(states.shape[0]):
            # state[i]: shape [input_dim] -> can pass raw numpy
            r = reward_model.compute_reward(states[i], actions[i].item(), step)
            rewards_list.append(r)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        # === Advantage ===
        # advantage = reward - V(s) (since single-step environment)
        advantages = rewards_tensor - values  # [batch]

        # === Losses ===
        # Policy loss (actor)
        policy_loss = -(advantages.detach() * log_probs).mean()

        # Value loss (critic): MSE(V(s), reward)
        value_loss = 0.5 * (advantages ** 2).mean()

        total_loss = policy_loss + value_loss

        # === Backprop ===
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            self.config.max_grad_norm
        )
        self.optimizer.step()

        return total_loss.item()

########################################
#               TRAINING
########################################

def train(config: Config) -> ActorCriticAgent:
    logger.info("Starting training with config: %s", config.__dict__)
    set_seed(config.seed)

    device = torch.device(config.device)

    # Instantiate the policy & value networks
    policy_net = PolicyNetwork(
        config.input_dim,
        config.output_dim,
        config.hidden_dim
    ).to(device)

    value_net = ValueNetwork(
        config.input_dim,
        config.hidden_dim
    ).to(device)

    # Create the agent
    agent = ActorCriticAgent(policy_net, value_net, config)

    # Reward model
    reward_model = RewardModel()

    # Directory to store checkpoints
    checkpoint_dir = Path(f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, config.num_epochs + 1):
            # Sample a batch of random states
            states = np.random.randn(config.batch_size, config.input_dim).astype(np.float32)

            # Perform one update step
            loss = agent.update(states, step=epoch, reward_model=reward_model)

            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{config.num_epochs} - Loss: {loss:.4f}")

            # (Optional) Store metrics
            agent.training_history.append({
                "epoch": epoch,
                "loss": float(loss)
            })

            # Save a checkpoint periodically
            if epoch % 50 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                agent.save(checkpoint_path)

        # Final save
        agent.save(checkpoint_dir / "final_model.pt")
        logger.info("Training completed successfully")
        return agent

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

########################################
#               MAIN
########################################

if __name__ == "__main__":
    try:
        config_path = "config.json"
        # Load config if file exists, else create a default one
        if Path(config_path).exists():
            config = Config.from_json(config_path)
        else:
            config = Config()
            config.save(config_path)

        # Train the agent
        agent = train(config)

        # Simple evaluation:
        # We'll generate a random state, have the agent pick an action distribution,
        # then sample an action and log its probability.
        test_state = np.random.randn(config.input_dim).astype(np.float32)
        test_state_tensor = torch.tensor(test_state, device=config.device).unsqueeze(0)
        with torch.no_grad():
            logits = agent.policy_network(test_state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            action_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        logger.info(f"\nEvaluation on a random test state:\n"
                    f"State: {test_state}\n"
                    f"Action Probabilities: {action_probs}\n"
                    f"Chosen Action: {action}")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)
