"""
Визуализация траекторий и метрик обучения.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from pathlib import Path


def plot_trajectory(
    agent_trajectory: np.ndarray,
    target_trajectory: np.ndarray,
    obstacles: List[Tuple[float, float, float]],
    area_size: float = 1.0,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Визуализация траектории агента и цели с препятствиями.
    
    Args:
        agent_trajectory: Массив позиций агента shape (N, 2)
        target_trajectory: Массив позиций цели shape (N, 2)
        obstacles: Список препятствий (x, y, radius)
        area_size: Размер области
        save_path: Путь для сохранения графика
        show: Показать ли график
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Рисуем препятствия
    for x, y, radius in obstacles:
        circle = plt.Circle((x, y), radius, color='gray', alpha=0.5)
        ax.add_patch(circle)
    
    # Рисуем траектории
    ax.plot(
        agent_trajectory[:, 0],
        agent_trajectory[:, 1],
        'b-',
        linewidth=2,
        label='Agent trajectory',
        alpha=0.7
    )
    ax.plot(
        target_trajectory[:, 0],
        target_trajectory[:, 1],
        'r-',
        linewidth=2,
        label='Target trajectory',
        alpha=0.7
    )
    
    # Начальные и конечные позиции
    ax.plot(agent_trajectory[0, 0], agent_trajectory[0, 1], 'bo', markersize=10, label='Agent start')
    ax.plot(agent_trajectory[-1, 0], agent_trajectory[-1, 1], 'bs', markersize=10, label='Agent end')
    ax.plot(target_trajectory[0, 0], target_trajectory[0, 1], 'ro', markersize=10, label='Target start')
    ax.plot(target_trajectory[-1, 0], target_trajectory[-1, 1], 'rs', markersize=10, label='Target end')
    
    # Настройки графика
    ax.set_xlim(-0.1, area_size + 0.1)
    ax.set_ylim(-0.1, area_size + 0.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X position', fontsize=12)
    ax.set_ylabel('Y position', fontsize=12)
    ax.set_title('Agent Interception Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_metrics(
    rewards: List[float],
    success_rate: Optional[List[float]] = None,
    episode_lengths: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    window_size: int = 50,
):
    """
    Визуализация метрик обучения.
    
    Args:
        rewards: Список наград по эпизодам
        success_rate: Список success rate (опционально)
        episode_lengths: Список длительностей эпизодов (опционально)
        save_path: Путь для сохранения графика
        show: Показать ли график
        window_size: Размер окна для скользящего среднего
    """
    num_plots = 1 + (success_rate is not None) + (episode_lengths is not None)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # График наград
    ax = axes[plot_idx]
    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw rewards')
    
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        ax.plot(
            episodes[window_size-1:],
            moving_avg,
            color='blue',
            linewidth=2,
            label=f'Moving avg (window={window_size})'
        )
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # График success rate
    if success_rate is not None:
        ax = axes[plot_idx]
        episodes = np.arange(len(success_rate))
        ax.plot(episodes, success_rate, color='green', linewidth=2)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # График длительности эпизодов
    if episode_lengths is not None:
        ax = axes[plot_idx]
        episodes = np.arange(len(episode_lengths))
        ax.plot(episodes, episode_lengths, alpha=0.3, color='red', label='Raw lengths')
        
        if len(episode_lengths) >= window_size:
            moving_avg = np.convolve(
                episode_lengths,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax.plot(
                episodes[window_size-1:],
                moving_avg,
                color='red',
                linewidth=2,
                label=f'Moving avg (window={window_size})'
            )
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Length', fontsize=12)
        ax.set_title('Episode Lengths', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training metrics plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results: dict,
    metric: str = 'reward',
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Сравнительный график для нескольких алгоритмов.
    
    Args:
        results: Словарь {algorithm_name: [values]}
        metric: Название метрики для заголовка
        save_path: Путь для сохранения
        show: Показать ли график
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in results.items():
        episodes = np.arange(len(values))
        plt.plot(episodes, values, label=name, linewidth=2, alpha=0.7)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'Algorithm Comparison: {metric.capitalize()}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

