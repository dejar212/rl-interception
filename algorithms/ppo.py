"""
PPO (Proximal Policy Optimization) алгоритм.
Адаптировано из CleanRL для задачи перехвата.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Optional


class ActorCritic(nn.Module):
    """Нейронная сеть для Actor-Critic архитектуры."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Общий feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor (policy) head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_mean: Среднее действие
            value: Оценка ценности состояния
        """
        features = self.shared(x)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Получить только оценку ценности состояния."""
        features = self.shared(x)
        return self.critic(features)
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Получить действие, log_prob, энтропию и ценность.
        
        Returns:
            action: Выбранное действие
            log_prob: Log вероятность действия
            entropy: Энтропия распределения
            value: Оценка ценности
        """
        action_mean, value = self.forward(x)
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value


class PPOAgent:
    """PPO агент для обучения с подкреплением."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # Создаем нейронную сеть
        self.actor_critic = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Предсказать действие для заданного наблюдения.
        
        Args:
            obs: Наблюдение из среды
            deterministic: Если True, возвращает среднее действие (без шума)
        
        Returns:
            Действие для выполнения
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_mean, _ = self.actor_critic(obs_tensor)
            
            if deterministic:
                action = action_mean
            else:
                action_std = self.actor_critic.actor_logstd.exp()
                dist = Normal(action_mean, action_std)
                action = dist.sample()
            
            return action.cpu().numpy()[0]
    
    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict:
        """
        Один шаг обучения PPO.
        
        Returns:
            Словарь с метриками обучения
        """
        # Forward pass
        _, log_probs_new, entropy, values = self.actor_critic.get_action_and_value(
            obs, actions
        )
        
        # Policy loss (PPO clipped objective)
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = 0.5 * ((values.squeeze() - returns) ** 2).mean()
        
        # Entropy loss (для exploration)
        entropy_loss = -entropy.mean()
        
        # Общий loss
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Метрики
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'clip_fraction': clip_fraction.item(),
        }
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Массив наград
            values: Массив оценок ценности
            dones: Массив флагов завершения эпизода
            next_value: Ценность следующего состояния
        
        Returns:
            advantages: Преимущества
            returns: Целевые ценности
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def save(self, path: str):
        """Сохранить модель."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Загрузить модель."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

