"""
Расширенные визуализации для анализа производительности.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path


def plot_heatmap(
    positions: np.ndarray,
    area_size: float = 1.0,
    bins: int = 50,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Agent Position Heatmap",
):
    """
    Создать тепловую карту посещаемости областей среды.
    
    Args:
        positions: Массив позиций shape (N, 2)
        area_size: Размер области
        bins: Количество bins для гистограммы
        save_path: Путь для сохранения
        show: Показать ли график
        title: Заголовок графика
    """
    if len(positions) == 0:
        print("Warning: No positions to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Создаем 2D гистограмму
    heatmap, xedges, yedges = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=bins,
        range=[[0, area_size], [0, area_size]]
    )
    
    # Отображаем тепловую карту
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        heatmap.T,
        extent=extent,
        origin='lower',
        cmap='hot',
        interpolation='gaussian',
        aspect='auto'
    )
    
    # Добавляем colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Count', fontsize=12)
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_box_comparison(
    data_dict: Dict[str, List[float]],
    metric_name: str = "Metric",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Создать box plot для сравнения алгоритмов.
    
    Args:
        data_dict: Словарь {algorithm_name: [values]}
        metric_name: Название метрики
        save_path: Путь для сохранения
        show: Показать ли график
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Подготовка данных
    data = [values for values in data_dict.values()]
    labels = list(data_dict.keys())
    
    # Создаем box plot
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        notch=True,
        showmeans=True,
    )
    
    # Раскрашиваем boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Comparison (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Box plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_violin_comparison(
    data_dict: Dict[str, List[float]],
    metric_name: str = "Metric",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Создать violin plot для сравнения алгоритмов.
    
    Args:
        data_dict: Словарь {algorithm_name: [values]}
        metric_name: Название метрики
        save_path: Путь для сохранения
        show: Показать ли график
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Подготовка данных для seaborn
    data_list = []
    for algo_name, values in data_dict.items():
        for val in values:
            data_list.append({'Algorithm': algo_name, metric_name: val})
    
    import pandas as pd
    df = pd.DataFrame(data_list)
    
    # Создаем violin plot
    sns.violinplot(
        data=df,
        x='Algorithm',
        y=metric_name,
        ax=ax,
        palette='Set3',
        inner='box',
    )
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Violin plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_grid(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Создать сетку графиков для нескольких метрик.
    
    Args:
        metrics_dict: Словарь {algorithm: {metric_name: value}}
        save_path: Путь для сохранения
        show: Показать ли график
    """
    # Определяем метрики
    if not metrics_dict:
        return
    
    all_metrics = set()
    for algo_metrics in metrics_dict.values():
        all_metrics.update(algo_metrics.keys())
    
    metrics_to_plot = [
        'mean_reward',
        'success_rate',
        'mean_episode_length',
        'mean_path_efficiency',
        'collision_rate',
        'mean_interception_time',
    ]
    
    # Фильтруем только существующие метрики
    metrics_to_plot = [m for m in metrics_to_plot if m in all_metrics]
    
    n_metrics = len(metrics_to_plot)
    if n_metrics == 0:
        return
    
    # Создаем сетку
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    algorithms = list(metrics_dict.keys())
    x_pos = np.arange(len(algorithms))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        values = [metrics_dict[algo].get(metric, 0) for algo in algorithms]
        
        bars = ax.bar(x_pos, values, color=plt.cm.Set3(np.linspace(0, 1, len(algorithms))))
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцах
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    # Скрываем лишние subplot'ы
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metric grid saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_radar(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Создать radar chart для сравнения производительности.
    
    Args:
        metrics_dict: Словарь {algorithm: {metric_name: value}}
        save_path: Путь для сохранения
        show: Показать ли график
    """
    metrics_to_plot = [
        'success_rate',
        'mean_path_efficiency',
        'mean_reward',
    ]
    
    # Нормализуем метрики в [0, 1]
    normalized = {}
    for metric in metrics_to_plot:
        values = [metrics_dict[algo].get(metric, 0) for algo in metrics_dict.keys()]
        min_val, max_val = min(values), max(values)
        
        if max_val > min_val:
            normalized[metric] = {
                algo: (metrics_dict[algo].get(metric, 0) - min_val) / (max_val - min_val)
                for algo in metrics_dict.keys()
            }
        else:
            normalized[metric] = {algo: 0.5 for algo in metrics_dict.keys()}
    
    # Создаем radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    for algo_idx, algo in enumerate(metrics_dict.keys()):
        values = [normalized[metric][algo] for metric in metrics_to_plot]
        values += values[:1]  # Замыкаем круг
        
        ax.plot(angles, values, 'o-', linewidth=2, label=algo)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    ax.set_ylim(0, 1)
    ax.set_title('Performance Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Radar chart saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

