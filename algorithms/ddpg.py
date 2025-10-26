"""
DDPG (Deep Deterministic Policy Gradient) implementation
Off-policy, deterministic actor-critic algorithm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from collections import deque
import random


class Actor(nn.Module):
    """DDPG Actor network"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.max_action


class Critic(nn.Module):
    """DDPG Critic network (Q-function)"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """DDPG Agent implementation"""
    
    def __init__(
        self,
        env,
        config: Dict,
        device: str = "cpu"
    ):
        self.env = env
        self.config = config
        self.device = device
        
        # Network dimensions
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        hidden_dim = config.get('hidden_dim', 256)
        max_action = float(env.action_space.high[0])
        
        # Actor networks
        self.actor = Actor(obs_dim, act_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        actor_lr = config.get('actor_learning_rate', 3e-4)
        critic_lr = config.get('critic_learning_rate', 3e-4)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        buffer_size = config.get('buffer_size', 1000000)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 256)
        self.learning_starts = config.get('learning_starts', 1000)
        self.train_freq = config.get('train_freq', 1)
        
        # Exploration noise
        self.expl_noise = config.get('exploration_noise', 0.1)
        self.max_action = max_action
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action with optional exploration noise"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
            
        if add_noise:
            noise = np.random.normal(0, self.expl_noise * self.max_action, size=action.shape)
            action = action + noise
            
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        current_q = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, total_timesteps: int):
        """Train the agent"""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_successes = []
        
        global_step = 0
        
        print(f"Starting DDPG training for {total_timesteps:,} timesteps")
        
        while global_step < total_timesteps:
            # Select action
            if global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs, add_noise=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            episode_length += 1
            global_step += 1
            
            # Update networks
            if global_step >= self.learning_starts and global_step % self.train_freq == 0:
                self.update()
            
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
        
        print(f"Training complete! Final reward: {np.mean(self.episode_rewards[-100:]):.2f}")
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation"""
        return self.select_action(obs, add_noise=not deterministic)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.success_rates = checkpoint.get('success_rates', [])
        print(f"Model loaded from {path}")

