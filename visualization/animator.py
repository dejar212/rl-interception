"""
Анимация эпизодов перехвата.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import List, Tuple, Optional
from pathlib import Path


def animate_episode(
    agent_trajectory: np.ndarray,
    target_trajectory: np.ndarray,
    obstacles: List[Tuple[float, float, float]],
    area_size: float = 1.0,
    save_path: Optional[str] = None,
    fps: int = 30,
    interval: int = 50,
):
    """
    Создать анимацию эпизода перехвата.
    
    Args:
        agent_trajectory: Массив позиций агента shape (N, 2)
        target_trajectory: Массив позиций цели shape (N, 2)
        obstacles: Список препятствий (x, y, radius)
        area_size: Размер области
        save_path: Путь для сохранения анимации (MP4 или GIF)
        fps: Кадров в секунду
        interval: Интервал между кадрами в мс
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Рисуем препятствия
    for x, y, radius in obstacles:
        circle = Circle((x, y), radius, color='gray', alpha=0.5, zorder=1)
        ax.add_patch(circle)
    
    # Настройки графика
    ax.set_xlim(-0.1, area_size + 0.1)
    ax.set_ylim(-0.1, area_size + 0.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X position', fontsize=12)
    ax.set_ylabel('Y position', fontsize=12)
    ax.set_title('Agent Interception Animation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Инициализация траекторий (линии)
    agent_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5, label='Agent path', zorder=2)
    target_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.5, label='Target path', zorder=2)
    
    # Инициализация агента и цели (точки)
    agent_point, = ax.plot([], [], 'bo', markersize=12, label='Agent', zorder=3)
    target_point, = ax.plot([], [], 'rs', markersize=12, label='Target', zorder=3)
    
    # Текст для отображения шага
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    ax.legend(loc='upper right')
    
    def init():
        """Инициализация анимации."""
        agent_line.set_data([], [])
        target_line.set_data([], [])
        agent_point.set_data([], [])
        target_point.set_data([], [])
        time_text.set_text('')
        return agent_line, target_line, agent_point, target_point, time_text
    
    def animate(frame):
        """Обновление кадра анимации."""
        # Обновляем траектории до текущего кадра
        agent_line.set_data(agent_trajectory[:frame+1, 0], agent_trajectory[:frame+1, 1])
        target_line.set_data(target_trajectory[:frame+1, 0], target_trajectory[:frame+1, 1])
        
        # Обновляем текущие позиции
        agent_point.set_data([agent_trajectory[frame, 0]], [agent_trajectory[frame, 1]])
        target_point.set_data([target_trajectory[frame, 0]], [target_trajectory[frame, 1]])
        
        # Обновляем текст
        time_text.set_text(f'Step: {frame}/{len(agent_trajectory)-1}')
        
        return agent_line, target_line, agent_point, target_point, time_text
    
    # Создаем анимацию
    n_frames = len(agent_trajectory)
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_frames,
        interval=interval,
        blit=True,
        repeat=True,
    )
    
    # Сохраняем или показываем
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.gif':
            print(f"Saving animation as GIF: {save_path}")
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.suffix in ['.mp4', '.avi']:
            print(f"Saving animation as video: {save_path}")
            try:
                anim.save(save_path, writer='ffmpeg', fps=fps)
            except Exception as e:
                print(f"Warning: Could not save video. Error: {e}")
                print("Trying to save as GIF instead...")
                gif_path = save_path.with_suffix('.gif')
                anim.save(gif_path, writer='pillow', fps=fps)
                print(f"Saved as GIF: {gif_path}")
        else:
            raise ValueError(f"Unsupported format: {save_path.suffix}. Use .gif, .mp4, or .avi")
        
        plt.close()
    else:
        plt.show()


def create_side_by_side_animation(
    trajectories_dict: dict,
    obstacles: List[Tuple[float, float, float]],
    area_size: float = 1.0,
    save_path: Optional[str] = None,
    fps: int = 30,
    interval: int = 50,
):
    """
    Создать анимацию с несколькими траекториями side-by-side.
    
    Args:
        trajectories_dict: Словарь {name: (agent_traj, target_traj)}
        obstacles: Список препятствий (x, y, radius)
        area_size: Размер области
        save_path: Путь для сохранения анимации
        fps: Кадров в секунду
        interval: Интервал между кадрами в мс
    """
    n_plots = len(trajectories_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(10*n_plots, 10))
    
    if n_plots == 1:
        axes = [axes]
    
    # Подготовка для каждого subplot
    plot_elements = []
    
    for idx, (name, (agent_traj, target_traj)) in enumerate(trajectories_dict.items()):
        ax = axes[idx]
        
        # Рисуем препятствия
        for x, y, radius in obstacles:
            circle = Circle((x, y), radius, color='gray', alpha=0.5, zorder=1)
            ax.add_patch(circle)
        
        # Настройки графика
        ax.set_xlim(-0.1, area_size + 0.1)
        ax.set_ylim(-0.1, area_size + 0.1)
        ax.set_aspect('equal')
        ax.set_xlabel('X position', fontsize=12)
        ax.set_ylabel('Y position', fontsize=12)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Элементы для анимации
        agent_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5, label='Agent', zorder=2)
        target_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.5, label='Target', zorder=2)
        agent_point, = ax.plot([], [], 'bo', markersize=12, zorder=3)
        target_point, = ax.plot([], [], 'rs', markersize=12, zorder=3)
        
        ax.legend(loc='upper right')
        
        plot_elements.append({
            'agent_traj': agent_traj,
            'target_traj': target_traj,
            'agent_line': agent_line,
            'target_line': target_line,
            'agent_point': agent_point,
            'target_point': target_point,
        })
    
    # Находим максимальную длину траектории
    max_frames = max(len(elem['agent_traj']) for elem in plot_elements)
    
    def init():
        """Инициализация анимации."""
        artists = []
        for elem in plot_elements:
            elem['agent_line'].set_data([], [])
            elem['target_line'].set_data([], [])
            elem['agent_point'].set_data([], [])
            elem['target_point'].set_data([], [])
            artists.extend([
                elem['agent_line'],
                elem['target_line'],
                elem['agent_point'],
                elem['target_point'],
            ])
        return artists
    
    def animate(frame):
        """Обновление кадра анимации."""
        artists = []
        for elem in plot_elements:
            # Учитываем разную длину траекторий
            current_frame = min(frame, len(elem['agent_traj']) - 1)
            
            # Обновляем траектории
            elem['agent_line'].set_data(
                elem['agent_traj'][:current_frame+1, 0],
                elem['agent_traj'][:current_frame+1, 1]
            )
            elem['target_line'].set_data(
                elem['target_traj'][:current_frame+1, 0],
                elem['target_traj'][:current_frame+1, 1]
            )
            
            # Обновляем текущие позиции
            elem['agent_point'].set_data(
                [elem['agent_traj'][current_frame, 0]],
                [elem['agent_traj'][current_frame, 1]]
            )
            elem['target_point'].set_data(
                [elem['target_traj'][current_frame, 0]],
                [elem['target_traj'][current_frame, 1]]
            )
            
            artists.extend([
                elem['agent_line'],
                elem['target_line'],
                elem['agent_point'],
                elem['target_point'],
            ])
        
        return artists
    
    # Создаем анимацию
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=max_frames,
        interval=interval,
        blit=True,
        repeat=True,
    )
    
    plt.tight_layout()
    
    # Сохраняем или показываем
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.gif':
            print(f"Saving comparison animation as GIF: {save_path}")
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.suffix in ['.mp4', '.avi']:
            print(f"Saving comparison animation as video: {save_path}")
            try:
                anim.save(save_path, writer='ffmpeg', fps=fps)
            except Exception as e:
                print(f"Warning: Could not save video. Error: {e}")
                print("Trying to save as GIF instead...")
                gif_path = save_path.with_suffix('.gif')
                anim.save(gif_path, writer='pillow', fps=fps)
                print(f"Saved as GIF: {gif_path}")
        
        plt.close()
    else:
        plt.show()

