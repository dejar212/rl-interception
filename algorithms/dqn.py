"""
DQN (Deep Q-Network) алгоритм для дискретных действий.
Адаптировано из CleanRL для задачи перехвата.

Примечание: DQN работает с дискретным action space, поэтому мы 
дискретизируем непрерывное пространство действий.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple
import copy


class QNetwork(nn.Module):
    """Q-Network для DQN."""
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class DQNAgent:
    """
    DQN агент для обучения с подкреплением.
    
    Использует дискретизацию действий для работы с непрерывным action space.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        tau: float = 1.0,
        target_update_frequency: int = 1000,
        buffer_size: int = 100000,
        n_discrete_actions: int = 5,  # Дискретизация по каждому измерению
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.target_update_frequency = target_update_frequency
        self.action_dim = action_dim
        self.n_discrete_actions = n_discrete_actions
        
        # Дискретизация action space
        # Для 2D действий: создаем комбинации дискретных значений
        self.discrete_values = np.linspace(-1, 1, n_discrete_actions)
        self.action_combinations = self._generate_action_combinations()
        self.n_actions = len(self.action_combinations)
        
        # Q-network
        self.q_network = QNetwork(obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Training step counter
        self.steps = 0
        
        # Replay buffer
        self.buffer_size = buffer_size
        self.buffer_ptr = 0
        self.buffer_full = False
        self.obs_buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs_buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros(buffer_size, dtype=np.int64)  # Индексы действий
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(buffer_size, dtype=np.float32)
    
    def _generate_action_combinations(self) -> np.ndarray:
        """Генерирует все комбинации дискретных действий."""
        if self.action_dim == 2:
            # Для 2D создаем сетку
            actions = []
            for a1 in self.discrete_values:
                for a2 in self.discrete_values:
                    actions.append([a1, a2])
            return np.array(actions)
        else:
            # Для произвольной размерности (менее эффективно)
            import itertools
            actions = list(itertools.product(self.discrete_values, repeat=self.action_dim))
            return np.array(actions)
    
    def _action_to_index(self, action: np.ndarray) -> int:
        """Конвертирует непрерывное действие в индекс ближайшего дискретного."""
        distances = np.linalg.norm(self.action_combinations - action, axis=1)
        return np.argmin(distances)
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Предсказать действие для заданного наблюдения."""
        # Epsilon-greedy exploration
        if not deterministic and np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.q_network(obs_tensor)
                action_idx = q_values.argmax().item()
        
        return self.action_combinations[action_idx]
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Сохранить переход в replay buffer."""
        action_idx = self._action_to_index(action)
        
        self.obs_buffer[self.buffer_ptr] = obs
        self.action_buffer[self.buffer_ptr] = action_idx
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
            'actions': torch.LongTensor(self.action_buffer[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.reward_buffer[indices]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs_buffer[indices]).to(self.device),
            'dones': torch.FloatTensor(self.done_buffer[indices]).to(self.device),
        }
    
    def train_step(self, batch_size: int) -> dict:
        """Один шаг обучения DQN."""
        self.steps += 1
        
        batch = self.sample_batch(batch_size)
        
        # Compute current Q-values
        current_q = self.q_network(batch['obs']).gather(1, batch['actions'].unsqueeze(-1)).squeeze(-1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(batch['next_obs']).max(dim=1)[0]
            target_q = batch['rewards'] + self.gamma * (1 - batch['dones']) * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network
        if self.steps % self.target_update_frequency == 0:
            if self.tau == 1.0:
                # Hard update
                self.target_network.load_state_dict(self.q_network.state_dict())
            else:
                # Soft update
                for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value': current_q.mean().item(),
        }
    
    def save(self, path: str):
        """Сохранить модель."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)
    
    def load(self, path: str):
        """Загрузить модель."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

