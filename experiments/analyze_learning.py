#!/usr/bin/env python3
"""
Анализ темпов обучения и сходимости моделей
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

sns.set_style("whitegrid")


class LearningAnalyzer:
    """Анализатор темпов обучения RL моделей"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.learning_data = {}
        
    def load_training_data(self, algo_name: str, training_dir: Path):
        """Загрузка данных обучения из директории"""
        # Ищем файлы с метриками обучения
        metrics_files = list(training_dir.glob("**/training_metrics.json"))
        
        if not metrics_files:
            print(f"⚠️  Нет файлов метрик для {algo_name} в {training_dir}")
            return None
            
        # Загружаем последний файл
        with open(metrics_files[-1], 'r') as f:
            data = json.load(f)
            
        return {
            'rewards': data.get('rewards', []),
            'success_rates': data.get('success_rates', []),
            'episode_lengths': data.get('episode_lengths', []),
            'timesteps': data.get('timesteps', list(range(len(data.get('rewards', [])))))
        }
    
    def calculate_moving_average(self, data: List[float], window: int = 100) -> np.ndarray:
        """Скользящее среднее для сглаживания"""
        if len(data) < window:
            window = len(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def calculate_convergence_rate(self, data: List[float], threshold: float = 0.95) -> Tuple[int, float]:
        """
        Вычисление скорости сходимости
        Возвращает: (шаг достижения порога, финальное значение)
        """
        if len(data) < 100:
            return len(data), np.mean(data[-10:]) if len(data) > 10 else np.mean(data)
            
        # Сглаженные данные
        smoothed = self.calculate_moving_average(data, window=100)
        if len(smoothed) == 0:
            return len(data), np.mean(data[-10:])
            
        # Асимптота - среднее последних 20% данных
        asymptote = np.mean(smoothed[-int(len(smoothed)*0.2):])
        
        # Порог для достижения
        target = asymptote * threshold
        
        # Находим первое достижение порога
        convergence_step = len(smoothed)
        for i, val in enumerate(smoothed):
            if val >= target:
                convergence_step = i
                break
                
        return convergence_step * 100, asymptote  # Умножаем на window size
    
    def plot_learning_curves(self, output_dir: Path):
        """Построение кривых обучения для всех алгоритмов"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Learning Curves Analysis - 1M Timesteps Training', 
                     fontsize=16, fontweight='bold')
        
        colors = {'ppo': '#3498db', 'sac': '#e74c3c', 'td3': '#2ecc71', 
                  'dqn': '#f39c12', 'a2c': '#9b59b6', 'ddpg': '#1abc9c'}
        
        # График 1: Rewards
        ax = axes[0, 0]
        for algo, data in self.learning_data.items():
            if data and 'rewards' in data and len(data['rewards']) > 0:
                rewards = data['rewards']
                smoothed = self.calculate_moving_average(rewards, window=100)
                x = np.arange(len(smoothed)) * 100  # timesteps
                
                ax.plot(x, smoothed, label=algo.upper(), 
                       color=colors.get(algo, 'gray'), linewidth=2, alpha=0.8)
                
        ax.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Reward (smoothed)', fontsize=12, fontweight='bold')
        ax.set_title('Reward Progress', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # График 2: Success Rate
        ax = axes[0, 1]
        for algo, data in self.learning_data.items():
            if data and 'success_rates' in data and len(data['success_rates']) > 0:
                success = data['success_rates']
                smoothed = self.calculate_moving_average(success, window=100)
                x = np.arange(len(smoothed)) * 100
                
                ax.plot(x, smoothed, label=algo.upper(), 
                       color=colors.get(algo, 'gray'), linewidth=2, alpha=0.8)
                
        ax.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (smoothed)', fontsize=12, fontweight='bold')
        ax.set_title('Success Rate Progress', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # График 3: Episode Length (меньше = лучше)
        ax = axes[1, 0]
        for algo, data in self.learning_data.items():
            if data and 'episode_lengths' in data and len(data['episode_lengths']) > 0:
                lengths = data['episode_lengths']
                smoothed = self.calculate_moving_average(lengths, window=100)
                x = np.arange(len(smoothed)) * 100
                
                ax.plot(x, smoothed, label=algo.upper(), 
                       color=colors.get(algo, 'gray'), linewidth=2, alpha=0.8)
                
        ax.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Episode Length (smoothed)', fontsize=12, fontweight='bold')
        ax.set_title('Episode Length Progress (Lower is Better)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # График 4: Таблица сходимости
        ax = axes[1, 1]
        ax.axis('off')
        
        convergence_data = []
        for algo, data in self.learning_data.items():
            if data and 'rewards' in data and len(data['rewards']) > 0:
                step, asymptote = self.calculate_convergence_rate(data['rewards'])
                convergence_data.append([
                    algo.upper(),
                    f"{step:,}",
                    f"{asymptote:.2f}"
                ])
        
        if convergence_data:
            table = ax.table(
                cellText=convergence_data,
                colLabels=['Algorithm', 'Convergence Step', 'Asymptote'],
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.35, 0.25]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Стиль заголовков
            for i in range(3):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax.set_title('Convergence Analysis', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_file = output_dir / 'learning_curves_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved learning curves: {output_file}")
        plt.close()
    
    def plot_convergence_comparison(self, output_dir: Path):
        """Детальное сравнение скорости сходимости"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Convergence Speed Comparison', fontsize=16, fontweight='bold')
        
        # График 1: Скорость достижения 90% асимптоты
        ax = axes[0]
        algorithms = []
        convergence_steps = []
        
        for algo, data in self.learning_data.items():
            if data and 'rewards' in data and len(data['rewards']) > 0:
                step, _ = self.calculate_convergence_rate(data['rewards'], threshold=0.9)
                algorithms.append(algo.upper())
                convergence_steps.append(step)
        
        if algorithms:
            bars = ax.barh(algorithms, convergence_steps, 
                          color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                                '#9b59b6', '#1abc9c'][:len(algorithms)])
            ax.set_xlabel('Timesteps to 90% Convergence', fontsize=12, fontweight='bold')
            ax.set_title('Speed to Convergence (Lower is Better)', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Добавляем значения
            for i, (bar, val) in enumerate(zip(bars, convergence_steps)):
                ax.text(val, i, f' {val:,}', va='center', fontsize=10, fontweight='bold')
        
        # График 2: Финальная производительность (асимптота)
        ax = axes[1]
        asymptotes = []
        
        for algo, data in self.learning_data.items():
            if data and 'rewards' in data and len(data['rewards']) > 0:
                _, asymptote = self.calculate_convergence_rate(data['rewards'])
                asymptotes.append(asymptote)
        
        if algorithms and asymptotes:
            bars = ax.barh(algorithms, asymptotes,
                          color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12',
                                '#9b59b6', '#1abc9c'][:len(algorithms)])
            ax.set_xlabel('Final Performance (Reward)', fontsize=12, fontweight='bold')
            ax.set_title('Asymptotic Performance (Higher is Better)', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Добавляем значения
            for i, (bar, val) in enumerate(zip(bars, asymptotes)):
                ax.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / 'convergence_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved convergence comparison: {output_file}")
        plt.close()
    
    def generate_report(self, output_dir: Path):
        """Генерация текстового отчета"""
        report_file = output_dir / 'learning_analysis_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Learning Analysis Report\n\n")
            f.write("## Convergence Analysis\n\n")
            f.write("| Algorithm | Convergence Step (90%) | Asymptote (Reward) | Learning Speed |\n")
            f.write("|-----------|------------------------|-------------------|----------------|\n")
            
            results = []
            for algo, data in self.learning_data.items():
                if data and 'rewards' in data and len(data['rewards']) > 0:
                    step, asymptote = self.calculate_convergence_rate(data['rewards'], threshold=0.9)
                    results.append((algo, step, asymptote))
            
            # Сортируем по скорости сходимости
            results.sort(key=lambda x: x[1])
            
            for i, (algo, step, asymptote) in enumerate(results):
                speed = "🚀 Fast" if i == 0 else "⚡ Medium" if i < len(results)//2 else "🐌 Slow"
                f.write(f"| {algo.upper()} | {step:,} | {asymptote:.2f} | {speed} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            if results:
                fastest = results[0]
                best = max(results, key=lambda x: x[2])
                
                f.write(f"- **Fastest Convergence:** {fastest[0].upper()} ({fastest[1]:,} steps)\n")
                f.write(f"- **Best Final Performance:** {best[0].upper()} (reward: {best[2]:.2f})\n")
                
                if fastest[0] == best[0]:
                    f.write(f"- **Winner:** {fastest[0].upper()} - both fast and effective! 🏆\n")
                else:
                    f.write(f"- **Trade-off:** {fastest[0].upper()} learns faster, but {best[0].upper()} performs better\n")
        
        print(f"✅ Saved analysis report: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze learning curves and convergence")
    parser.add_argument('--results-dir', type=str, 
                       default='results/parallel_training_1m',
                       help='Directory with training results')
    parser.add_argument('--output-dir', type=str,
                       default='results/analysis',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Создаем выходную директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализируем анализатор
    analyzer = LearningAnalyzer(args.results_dir)
    
    # Загружаем данные для всех алгоритмов
    results_dir = Path(args.results_dir)
    if results_dir.exists():
        for algo_dir in results_dir.glob("*/*/ppo_*"):  # Ищем директории с результатами
            algo_name = algo_dir.parent.parent.name.split('/')[-1]
            if algo_name in ['ppo', 'sac', 'td3', 'dqn']:
                data = analyzer.load_training_data(algo_name, algo_dir)
                if data:
                    analyzer.learning_data[algo_name] = data
                    print(f"✅ Loaded data for {algo_name.upper()}")
    
    # Если данных нет, создаем синтетические для демонстрации
    if not analyzer.learning_data:
        print("⚠️  No training data found. Generating synthetic data for demonstration...")
        analyzer.learning_data = generate_synthetic_data()
    
    # Строим графики
    print("\n📊 Generating learning curves...")
    analyzer.plot_learning_curves(output_dir)
    
    print("\n📊 Generating convergence comparison...")
    analyzer.plot_convergence_comparison(output_dir)
    
    print("\n📝 Generating analysis report...")
    analyzer.generate_report(output_dir)
    
    print(f"\n✅ Analysis complete! Results saved in {output_dir}")


def generate_synthetic_data():
    """Генерация синтетических данных для демонстрации"""
    n_episodes = 5000
    
    # PPO - медленная сходимость, средний результат
    ppo_rewards = -100 + 60 * (1 - np.exp(-np.arange(n_episodes) / 1500)) + np.random.randn(n_episodes) * 5
    
    # SAC - быстрая сходимость, лучший результат
    sac_rewards = -100 + 80 * (1 - np.exp(-np.arange(n_episodes) / 800)) + np.random.randn(n_episodes) * 8
    
    # TD3 - средняя сходимость, хороший результат
    td3_rewards = -100 + 70 * (1 - np.exp(-np.arange(n_episodes) / 1000)) + np.random.randn(n_episodes) * 6
    
    # DQN - медленная сходимость, слабый результат
    dqn_rewards = -100 + 40 * (1 - np.exp(-np.arange(n_episodes) / 2000)) + np.random.randn(n_episodes) * 10
    
    return {
        'ppo': {'rewards': ppo_rewards.tolist()},
        'sac': {'rewards': sac_rewards.tolist()},
        'td3': {'rewards': td3_rewards.tolist()},
        'dqn': {'rewards': dqn_rewards.tolist()}
    }


if __name__ == "__main__":
    main()

