"""
A2C (Advantage Actor-Critic) implementation
On-policy algorithm with parallel actors
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple


class A2CNetwork(nn.Module):
    """Actor-Critic network for A2C"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (action_mean, action_std, value)
        """
        features = self.shared(obs)
        
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, obs: torch.Tensor, action=None):
        """Get action and value for training"""
        action_mean, action_std, value = self.forward(obs)
        
        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value


class A2CAgent:
    """A2C Agent implementation"""
    
    def __init__(
        self,
        env,
        config: Dict,
        device: str = "cpu"
    ):
        self.env = env
        self.config = config
        self.device = device
        
        # Network
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        hidden_dim = config.get('hidden_dim', 256)
        
        self.network = A2CNetwork(obs_dim, act_dim, hidden_dim).to(device)
        
        # Optimizer
        lr = config.get('learning_rate', 7e-4)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        self.num_steps = config.get('num_steps', 5)  # Steps per update (shorter than PPO)
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
    def train(self, total_timesteps: int):
        """Train the agent"""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_successes = []
        
        global_step = 0
        
        # Storage for rollout
        obs_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        values_buffer = []
        rewards_buffer = []
        dones_buffer = []
        
        print(f"Starting A2C training for {total_timesteps:,} timesteps")
        
        while global_step < total_timesteps:
            # Collect rollout
            for step in range(self.num_steps):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
                    
                action_np = action.cpu().numpy()[0]
                
                # Clip action
                action_np = np.clip(action_np, 
                                   self.env.action_space.low,
                                   self.env.action_space.high)
                
                # Store
                obs_buffer.append(obs)
                actions_buffer.append(action_np)
                log_probs_buffer.append(log_prob.item())
                values_buffer.append(value.item())
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated
                
                rewards_buffer.append(reward)
                dones_buffer.append(done)
                
                episode_reward += reward
                episode_length += 1
                global_step += 1
                
                obs = next_obs
                
                if done:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    success = info.get('success', False)
                    episode_successes.append(float(success))
                    
                    if len(episode_successes) >= 100:
                        self.success_rates.append(np.mean(episode_successes[-100:]))
                    
                    if global_step % 10000 == 0:
                        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                        success_rate = np.mean(episode_successes[-100:]) if len(episode_successes) >= 100 else np.mean(episode_successes)
                        print(f"Step {global_step:,}/{total_timesteps:,} | "
                              f"Reward: {avg_reward:.2f} | "
                              f"Success: {success_rate*100:.1f}%")
                    
                    obs, _ = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                
                if global_step >= total_timesteps:
                    break
            
            # Compute advantages and returns
            if len(obs_buffer) > 0:
                # Get value of last state
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    _, _, _, next_value = self.network.get_action_and_value(obs_tensor)
                    next_value = next_value.item()
                
                # Compute returns
                returns = []
                R = next_value
                for r, done in zip(reversed(rewards_buffer), reversed(dones_buffer)):
                    if done:
                        R = 0
                    R = r + self.gamma * R
                    returns.insert(0, R)
                
                # Convert to tensors
                obs_tensor = torch.FloatTensor(np.array(obs_buffer)).to(self.device)
                actions_tensor = torch.FloatTensor(np.array(actions_buffer)).to(self.device)
                returns_tensor = torch.FloatTensor(returns).to(self.device)
                values_tensor = torch.FloatTensor(values_buffer).to(self.device)
                
                # Compute advantages
                advantages = returns_tensor - values_tensor
                
                # Get current predictions
                _, log_probs, entropy, values = self.network.get_action_and_value(
                    obs_tensor, actions_tensor)
                
                # Policy loss
                policy_loss = -(log_probs * advantages.detach()).mean()
                
                # Value loss
                value_loss = 0.5 * ((values.squeeze() - returns_tensor) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Clear buffers
                obs_buffer = []
                actions_buffer = []
                log_probs_buffer = []
                values_buffer = []
                rewards_buffer = []
                dones_buffer = []
        
        print(f"Training complete! Final reward: {np.mean(self.episode_rewards[-100:]):.2f}")
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_std, _ = self.network(obs_tensor)
            
            if deterministic:
                action = action_mean
            else:
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
        
        action_np = action.cpu().numpy()[0]
        return np.clip(action_np, self.env.action_space.low, self.env.action_space.high)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.success_rates = checkpoint.get('success_rates', [])
        print(f"Model loaded from {path}")

