"""
Расширенные метрики для анализа производительности агентов.
"""
import numpy as np
from typing import List, Dict, Any
import json
import csv
from pathlib import Path


class MetricsCalculator:
    """Класс для вычисления расширенных метрик."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сброс всех накопленных данных."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_flags = []
        self.collision_flags = []
        self.collision_counts = []
        self.interception_times = []
        self.path_lengths = []
        self.direct_distances = []
        self.all_positions = []  # Для тепловых карт
    
    def add_episode(
        self,
        reward: float,
        length: int,
        success: bool,
        collision: bool,
        collision_count: int,
        interception_time: float = None,
        agent_trajectory: np.ndarray = None,
        target_start: np.ndarray = None,
        target_end: np.ndarray = None,
    ):
        """
        Добавить результаты эпизода.
        
        Args:
            reward: Суммарная награда
            length: Длина эпизода (шагов)
            success: Флаг успеха
            collision: Флаг столкновения
            collision_count: Количество столкновений
            interception_time: Время до перехвата (если успех)
            agent_trajectory: Траектория агента для расчета длины пути
            target_start: Начальная позиция цели
            target_end: Конечная позиция цели
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_flags.append(success)
        self.collision_flags.append(collision)
        self.collision_counts.append(collision_count)
        
        if success and interception_time is not None:
            self.interception_times.append(interception_time)
        
        # Вычисляем длину пути и эффективность
        if agent_trajectory is not None and len(agent_trajectory) > 1:
            path_length = self._calculate_path_length(agent_trajectory)
            self.path_lengths.append(path_length)
            
            # Сохраняем позиции для тепловой карты
            self.all_positions.extend(agent_trajectory.tolist())
            
            # Вычисляем прямое расстояние
            if target_start is not None and target_end is not None:
                direct_dist = np.linalg.norm(target_end - target_start)
                self.direct_distances.append(direct_dist)
    
    def _calculate_path_length(self, trajectory: np.ndarray) -> float:
        """Вычислить длину пройденного пути."""
        if len(trajectory) < 2:
            return 0.0
        
        differences = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(differences, axis=1)
        return float(np.sum(distances))
    
    def calculate_path_efficiency(self) -> List[float]:
        """Вычислить эффективность пути (прямое расстояние / пройденное)."""
        if not self.path_lengths or not self.direct_distances:
            return []
        
        efficiencies = []
        for path_len, direct_dist in zip(self.path_lengths, self.direct_distances):
            if path_len > 0:
                efficiency = direct_dist / path_len
                efficiencies.append(min(efficiency, 1.0))  # Не больше 1
            else:
                efficiencies.append(0.0)
        
        return efficiencies
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Получить сводку всех метрик.
        
        Returns:
            Словарь с метриками
        """
        n_episodes = len(self.episode_rewards)
        
        if n_episodes == 0:
            return {}
        
        path_efficiencies = self.calculate_path_efficiency()
        
        summary = {
            'total_episodes': n_episodes,
            
            # Reward метрики
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'median_reward': float(np.median(self.episode_rewards)),
            
            # Success метрики
            'success_rate': float(np.mean(self.success_flags)),
            'total_successes': int(np.sum(self.success_flags)),
            
            # Collision метрики
            'collision_rate': float(np.mean(self.collision_flags)),
            'mean_collisions_per_episode': float(np.mean(self.collision_counts)),
            'total_collisions': int(np.sum(self.collision_counts)),
            
            # Episode length метрики
            'mean_episode_length': float(np.mean(self.episode_lengths)),
            'std_episode_length': float(np.std(self.episode_lengths)),
            'min_episode_length': int(np.min(self.episode_lengths)),
            'max_episode_length': int(np.max(self.episode_lengths)),
        }
        
        # Interception time (только для успешных)
        if self.interception_times:
            summary['mean_interception_time'] = float(np.mean(self.interception_times))
            summary['std_interception_time'] = float(np.std(self.interception_times))
            summary['min_interception_time'] = float(np.min(self.interception_times))
        
        # Path metrics
        if self.path_lengths:
            summary['mean_path_length'] = float(np.mean(self.path_lengths))
            summary['std_path_length'] = float(np.std(self.path_lengths))
        
        # Path efficiency
        if path_efficiencies:
            summary['mean_path_efficiency'] = float(np.mean(path_efficiencies))
            summary['std_path_efficiency'] = float(np.std(path_efficiencies))
        
        return summary
    
    def save_to_json(self, path: str):
        """Сохранить метрики в JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'summary': self.get_summary(),
            'episodes': {
                'rewards': self.episode_rewards,
                'lengths': self.episode_lengths,
                'successes': self.success_flags,
                'collisions': self.collision_flags,
                'collision_counts': self.collision_counts,
                'interception_times': self.interception_times,
                'path_lengths': self.path_lengths,
                'path_efficiencies': self.calculate_path_efficiency(),
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_to_csv(self, path: str):
        """Сохранить данные эпизодов в CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path_efficiencies = self.calculate_path_efficiency()
        
        # Выравниваем списки (дополняем None где нужно)
        max_len = len(self.episode_rewards)
        interception_times_padded = self.interception_times + [None] * (max_len - len(self.interception_times))
        path_lengths_padded = self.path_lengths + [None] * (max_len - len(self.path_lengths))
        path_efficiencies_padded = path_efficiencies + [None] * (max_len - len(path_efficiencies))
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'episode',
                'reward',
                'length',
                'success',
                'collision',
                'collision_count',
                'interception_time',
                'path_length',
                'path_efficiency',
            ])
            
            # Data
            for i in range(max_len):
                writer.writerow([
                    i + 1,
                    self.episode_rewards[i],
                    self.episode_lengths[i],
                    self.success_flags[i],
                    self.collision_flags[i],
                    self.collision_counts[i],
                    interception_times_padded[i] if interception_times_padded[i] is not None else '',
                    path_lengths_padded[i] if path_lengths_padded[i] is not None else '',
                    path_efficiencies_padded[i] if path_efficiencies_padded[i] is not None else '',
                ])
    
    def get_positions_array(self) -> np.ndarray:
        """Получить массив всех посещенных позиций."""
        if not self.all_positions:
            return np.array([])
        return np.array(self.all_positions)


def calculate_convergence_speed(rewards: List[float], success_threshold: float = 0.8, window: int = 50) -> int:
    """
    Вычислить скорость обучения (эпизоды до конвергенции).
    
    Args:
        rewards: Список наград
        success_threshold: Порог success rate для конвергенции
        window: Размер окна для усреднения
    
    Returns:
        Номер эпизода конвергенции или -1 если не достигнуто
    """
    if len(rewards) < window:
        return -1
    
    # Нормализуем награды в [0, 1]
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    if max_reward == min_reward:
        return -1
    
    normalized = [(r - min_reward) / (max_reward - min_reward) for r in rewards]
    
    # Проверяем скользящее среднее
    for i in range(window, len(normalized)):
        window_avg = np.mean(normalized[i-window:i])
        if window_avg >= success_threshold:
            return i
    
    return -1

