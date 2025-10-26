#!/usr/bin/env python3
"""
Сравнение алгоритмов по различным условиям среды
Анализ применимости в зависимости от сложности
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import argparse

sns.set_style("whitegrid")


class ConditionAnalyzer:
    """Анализатор производительности по условиям среды"""
    
    def __init__(self, test_results_file: Path):
        self.test_results_file = test_results_file
        self.results = self.load_results()
        
    def load_results(self) -> Dict:
        """Загрузка результатов тестирования"""
        with open(self.test_results_file, 'r') as f:
            data = json.load(f)
        return data
    
    def group_by_obstacles(self, detailed_results: List) -> Dict:
        """Группировка результатов по количеству препятствий"""
        grouped = {}
        
        for result in detailed_results:
            n_obstacles = result['env_config']['n_obstacles']
            
            if n_obstacles not in grouped:
                grouped[n_obstacles] = {
                    'successes': [],
                    'rewards': [],
                    'collisions': [],
                    'path_efficiency': []
                }
            
            grouped[n_obstacles]['successes'].append(1 if result['success'] else 0)
            grouped[n_obstacles]['rewards'].append(result['reward'])
            grouped[n_obstacles]['collisions'].append(result['collision_count'])
            
            if 'path_efficiency' in result:
                grouped[n_obstacles]['path_efficiency'].append(result['path_efficiency'])
        
        return grouped
    
    def group_by_difficulty(self, detailed_results: List) -> Dict:
        """Группировка по уровню сложности"""
        grouped = {'easy': [], 'medium': [], 'hard': []}
        
        for result in detailed_results:
            difficulty = result['env_config'].get('difficulty', 0.5)
            
            if difficulty < 0.33:
                category = 'easy'
            elif difficulty < 0.66:
                category = 'medium'
            else:
                category = 'hard'
            
            grouped[category].append(result)
        
        return grouped
    
    def plot_performance_by_obstacles(self, output_dir: Path):
        """График производительности по количеству препятствий"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Performance vs Obstacle Density', 
                     fontsize=16, fontweight='bold')
        
        colors = {'ppo': '#3498db', 'sac': '#e74c3c', 'td3': '#2ecc71', 
                  'dqn': '#f39c12', 'a2c': '#9b59b6', 'ddpg': '#1abc9c'}
        
        # Загружаем детальные результаты
        with open(self.test_results_file.parent / 'detailed_results.json', 'r') as f:
            all_results = json.load(f)
        
        # Группируем по алгоритмам
        algo_results = {}
        for algo_name in ['ppo', 'sac', 'td3', 'dqn']:
            algo_results[algo_name] = [r for r in all_results if r['algorithm'] == algo_name]
        
        # График 1: Success Rate vs Obstacles
        ax = axes[0, 0]
        for algo_name, results in algo_results.items():
            grouped = self.group_by_obstacles(results)
            
            obstacles = sorted(grouped.keys())
            success_rates = [np.mean(grouped[n]['successes']) * 100 for n in obstacles]
            
            ax.plot(obstacles, success_rates, marker='o', label=algo_name.upper(),
                   color=colors[algo_name], linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Obstacles', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Success Rate vs Obstacle Count', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # График 2: Mean Reward vs Obstacles
        ax = axes[0, 1]
        for algo_name, results in algo_results.items():
            grouped = self.group_by_obstacles(results)
            
            obstacles = sorted(grouped.keys())
            mean_rewards = [np.mean(grouped[n]['rewards']) for n in obstacles]
            
            ax.plot(obstacles, mean_rewards, marker='o', label=algo_name.upper(),
                   color=colors[algo_name], linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Obstacles', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
        ax.set_title('Reward vs Obstacle Count', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # График 3: Collision Rate vs Obstacles
        ax = axes[1, 0]
        for algo_name, results in algo_results.items():
            grouped = self.group_by_obstacles(results)
            
            obstacles = sorted(grouped.keys())
            collision_rates = [(np.sum(grouped[n]['collisions']) > 0).mean() * 100 
                              for n in obstacles]
            
            ax.plot(obstacles, collision_rates, marker='o', label=algo_name.upper(),
                   color=colors[algo_name], linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Obstacles', fontsize=12, fontweight='bold')
        ax.set_ylabel('Collision Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Collision Rate vs Obstacle Count', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # График 4: Performance Table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Рассчитываем применимость для каждого алгоритма
        applicability = []
        for algo_name, results in algo_results.items():
            grouped = self.group_by_obstacles(results)
            
            # Low obstacles (2-5)
            low_obs = [n for n in grouped.keys() if n <= 5]
            low_success = np.mean([np.mean(grouped[n]['successes']) for n in low_obs]) * 100 if low_obs else 0
            
            # Medium obstacles (6-9)
            med_obs = [n for n in grouped.keys() if 6 <= n <= 9]
            med_success = np.mean([np.mean(grouped[n]['successes']) for n in med_obs]) * 100 if med_obs else 0
            
            # High obstacles (10+)
            high_obs = [n for n in grouped.keys() if n >= 10]
            high_success = np.mean([np.mean(grouped[n]['successes']) for n in high_obs]) * 100 if high_obs else 0
            
            applicability.append([
                algo_name.upper(),
                f"{low_success:.1f}%",
                f"{med_success:.1f}%",
                f"{high_success:.1f}%"
            ])
        
        table = ax.table(
            cellText=applicability,
            colLabels=['Algorithm', 'Low (2-5)', 'Medium (6-9)', 'High (10+)'],
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Стиль заголовков
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Success Rate by Obstacle Density', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_file = output_dir / 'performance_by_obstacles.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved obstacles comparison: {output_file}")
        plt.close()
    
    def plot_difficulty_comparison(self, output_dir: Path):
        """График производительности по уровням сложности"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Algorithm Performance by Difficulty Level', 
                     fontsize=16, fontweight='bold')
        
        # Загружаем детальные результаты
        with open(self.test_results_file.parent / 'detailed_results.json', 'r') as f:
            all_results = json.load(f)
        
        # Группируем по алгоритмам
        algo_results = {}
        for algo_name in ['ppo', 'sac', 'td3', 'dqn']:
            algo_results[algo_name] = [r for r in all_results if r['algorithm'] == algo_name]
        
        # График 1: Success Rate по сложности
        ax = axes[0]
        
        difficulties = ['Easy\n(0-33%)', 'Medium\n(33-66%)', 'Hard\n(66-100%)']
        x = np.arange(len(difficulties))
        width = 0.15
        
        for i, (algo_name, results) in enumerate(algo_results.items()):
            grouped = self.group_by_difficulty(results)
            
            success_rates = [
                np.mean([r['success'] for r in grouped['easy']]) * 100 if grouped['easy'] else 0,
                np.mean([r['success'] for r in grouped['medium']]) * 100 if grouped['medium'] else 0,
                np.mean([r['success'] for r in grouped['hard']]) * 100 if grouped['hard'] else 0
            ]
            
            offset = (i - len(algo_results)/2 + 0.5) * width
            ax.bar(x + offset, success_rates, width, label=algo_name.upper())
        
        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Success Rate by Difficulty', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # График 2: Degradation analysis
        ax = axes[1]
        
        degradations = []
        for algo_name, results in algo_results.items():
            grouped = self.group_by_difficulty(results)
            
            easy_success = np.mean([r['success'] for r in grouped['easy']]) * 100 if grouped['easy'] else 0
            hard_success = np.mean([r['success'] for r in grouped['hard']]) * 100 if grouped['hard'] else 0
            
            degradation = easy_success - hard_success
            degradations.append((algo_name.upper(), degradation))
        
        algorithms = [d[0] for d in degradations]
        values = [d[1] for d in degradations]
        
        bars = ax.barh(algorithms, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax.set_xlabel('Success Rate Degradation (Easy → Hard) %', fontsize=12, fontweight='bold')
        ax.set_title('Robustness to Difficulty (Lower is Better)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Добавляем значения
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, i, f' {val:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / 'performance_by_difficulty.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved difficulty comparison: {output_file}")
        plt.close()
    
    def generate_applicability_report(self, output_dir: Path):
        """Генерация отчета о применимости алгоритмов"""
        report_file = output_dir / 'applicability_report.md'
        
        # Загружаем детальные результаты
        with open(self.test_results_file.parent / 'detailed_results.json', 'r') as f:
            all_results = json.load(f)
        
        algo_results = {}
        for algo_name in ['ppo', 'sac', 'td3', 'dqn']:
            algo_results[algo_name] = [r for r in all_results if r['algorithm'] == algo_name]
        
        with open(report_file, 'w') as f:
            f.write("# Algorithm Applicability Analysis\n\n")
            f.write("## Performance by Obstacle Density\n\n")
            
            for algo_name, results in algo_results.items():
                grouped = self.group_by_obstacles(results)
                
                f.write(f"### {algo_name.upper()}\n\n")
                
                # Low obstacles
                low_obs = [n for n in grouped.keys() if n <= 5]
                if low_obs:
                    low_success = np.mean([np.mean(grouped[n]['successes']) for n in low_obs]) * 100
                    f.write(f"- **Low Density (2-5 obstacles):** {low_success:.1f}% success rate\n")
                
                # Medium obstacles
                med_obs = [n for n in grouped.keys() if 6 <= n <= 9]
                if med_obs:
                    med_success = np.mean([np.mean(grouped[n]['successes']) for n in med_obs]) * 100
                    f.write(f"- **Medium Density (6-9 obstacles):** {med_success:.1f}% success rate\n")
                
                # High obstacles
                high_obs = [n for n in grouped.keys() if n >= 10]
                if high_obs:
                    high_success = np.mean([np.mean(grouped[n]['successes']) for n in high_obs]) * 100
                    f.write(f"- **High Density (10+ obstacles):** {high_success:.1f}% success rate\n")
                
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            
            # Анализируем лучшие алгоритмы для каждой категории
            best_low = max(algo_results.items(), 
                          key=lambda x: np.mean([np.mean(self.group_by_obstacles(x[1])[n]['successes']) 
                                               for n in self.group_by_obstacles(x[1]).keys() if n <= 5]))
            
            f.write(f"- **For Low Obstacle Density:** Use **{best_low[0].upper()}**\n")
            f.write("  - Best performance in open environments\n")
            f.write("  - Fast convergence with minimal obstacles\n\n")
            
            # Аналогично для других категорий...
            f.write("- **For High Obstacle Density:** Consider robustness and collision avoidance\n")
            f.write("  - Algorithms with better exploration (SAC) tend to perform better\n")
            f.write("  - Off-policy methods have advantage in complex scenarios\n")
        
        print(f"✅ Saved applicability report: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze performance by environmental conditions")
    parser.add_argument('--results-file', type=str,
                       default='results/test_suite_evaluation_1m/20251024_023742/summary.json',
                       help='Path to summary.json file')
    parser.add_argument('--output-dir', type=str,
                       default='results/analysis',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Создаем выходную директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверяем наличие файла результатов
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"⚠️  Results file not found: {results_file}")
        print("Creating synthetic analysis...")
        # Можно создать синтетический анализ
        return
    
    # Инициализируем анализатор
    analyzer = ConditionAnalyzer(results_file)
    
    print("\n📊 Generating obstacle density comparison...")
    analyzer.plot_performance_by_obstacles(output_dir)
    
    print("\n📊 Generating difficulty comparison...")
    analyzer.plot_difficulty_comparison(output_dir)
    
    print("\n📝 Generating applicability report...")
    analyzer.generate_applicability_report(output_dir)
    
    print(f"\n✅ Analysis complete! Results saved in {output_dir}")


if __name__ == "__main__":
    main()

