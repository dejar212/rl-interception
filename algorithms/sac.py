"""
SAC (Soft Actor-Critic) алгоритм.
Адаптировано из CleanRL для задачи перехвата.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional
import copy


class Actor(nn.Module):
    """Actor network для SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Action rescaling
        self.action_scale = 1.0
        self.action_bias = 0.0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.net(x)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            mean: Mean of distribution
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Enforcing action bounds
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class Critic(nn.Module):
    """Critic network (Q-function) для SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(torch.cat([x, action], dim=-1))


class SACAgent:
    """SAC агент для обучения с подкреплением."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        buffer_size: int = 1000000,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Actor network
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        # Critic networks (double Q-learning)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Target critics
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
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
            if deterministic:
                _, _, action = self.actor.get_action(obs_tensor)
            else:
                action, _, _ = self.actor.get_action(obs_tensor)
            return action.cpu().numpy()[0]
    
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
        """Один шаг обучения SAC."""
        batch = self.sample_batch(batch_size)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action(batch['next_obs'])
            q1_next = self.critic1_target(batch['next_obs'], next_actions)
            q2_next = self.critic2_target(batch['next_obs'], next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = batch['rewards'].unsqueeze(-1) + self.gamma * (1 - batch['dones'].unsqueeze(-1)) * min_q_next
        
        q1 = self.critic1(batch['obs'], batch['actions'])
        q2 = self.critic2(batch['obs'], batch['actions'])
        
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actions, log_probs, _ = self.actor.get_action(batch['obs'])
        q1_pi = self.critic1(batch['obs'], actions)
        q2_pi = self.critic2(batch['obs'], actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature parameter)
        alpha_loss = torch.tensor(0.0)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
        }
    
    def save(self, path: str):
        """Сохранить модель."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
        }, path)
    
    def load(self, path: str):
        """Загрузить модель."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']

