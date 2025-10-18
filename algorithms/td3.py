"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) алгоритм.
Адаптировано из CleanRL для задачи перехвата.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple
import copy


class Actor(nn.Module):
    """Actor network для TD3."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        
        self.max_action = max_action
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x) * self.max_action


class Critic(nn.Module):
    """Critic network (Q-function) для TD3."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both Q networks."""
        xu = torch.cat([x, action], dim=-1)
        return self.q1(xu), self.q2(xu)
    
    def q1_forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass for Q1 only."""
        xu = torch.cat([x, action], dim=-1)
        return self.q1(xu)


class TD3Agent:
    """TD3 агент для обучения с подкреплением."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        buffer_size: int = 1000000,
        max_action: float = 1.0,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.max_action = max_action
        
        # Actor network
        self.actor = Actor(obs_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        # Critic network
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Training step counter
        self.total_it = 0
        
        # Replay buffer
        self.buffer_size = buffer_size
        self.buffer_ptr = 0
        self.buffer_full = False
        self.obs_buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs_buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(buffer_size, dtype=np.float32)
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Предсказать действие для заданного наблюдения."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor).cpu().numpy()[0]
            
            if not deterministic:
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                action = np.clip(action + noise, -self.max_action, self.max_action)
            
            return action
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Сохранить переход в replay buffer."""
        self.obs_buffer[self.buffer_ptr] = obs
        self.action_buffer[self.buffer_ptr] = action
        self.reward_buffer[self.buffer_ptr] = reward
        self.next_obs_buffer[self.buffer_ptr] = next_obs
        self.done_buffer[self.buffer_ptr] = done
        
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
        if self.buffer_ptr == 0:
            self.buffer_full = True
    
    def sample_batch(self, batch_size: int) -> dict:
        """Сэмплировать батч из replay buffer."""
        max_idx = self.buffer_size if self.buffer_full else self.buffer_ptr
        indices = np.random.randint(0, max_idx, size=batch_size)
        
        return {
            'obs': torch.FloatTensor(self.obs_buffer[indices]).to(self.device),
            'actions': torch.FloatTensor(self.action_buffer[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.reward_buffer[indices]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs_buffer[indices]).to(self.device),
            'dones': torch.FloatTensor(self.done_buffer[indices]).to(self.device),
        }
    
    def train_step(self, batch_size: int) -> dict:
        """Один шаг обучения TD3."""
        self.total_it += 1
        
        batch = self.sample_batch(batch_size)
        
        # Update critics
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(batch['actions']) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(batch['next_obs']) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(batch['next_obs'], next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = batch['rewards'].unsqueeze(-1) + self.gamma * (1 - batch['dones'].unsqueeze(-1)) * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(batch['obs'], batch['actions'])
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        
        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(batch['obs'], self.actor(batch['obs'])).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
        }
    
    def save(self, path: str):
        """Сохранить модель."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
        }, path)
    
    def load(self, path: str):
        """Загрузить модель."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_it = checkpoint['total_it']

