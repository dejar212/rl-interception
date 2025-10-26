#!/usr/bin/env python3
"""
Быстрый анализ производительности по условиям
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def load_results(results_file):
    """Загрузка результатов"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    with open(results_file.parent / 'detailed_results.json', 'r') as f:
        detailed = json.load(f)
    
    return data, detailed


def plot_performance_summary(summary_data, output_dir):
    """График общей производительности"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Algorithm Performance Summary (1M Timesteps)', 
                 fontsize=16, fontweight='bold')
    
    algorithms = list(summary_data.keys())
    colors = {'ppo': '#3498db', 'sac': '#e74c3c', 'td3': '#2ecc71', 'dqn': '#f39c12'}
    
    # Success Rate
    ax = axes[0, 0]
    success_rates = [summary_data[algo]['success_rate'] * 100 for algo in algorithms]
    bars = ax.bar(algorithms, success_rates, color=[colors[a] for a in algorithms])
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Mean Reward
    ax = axes[0, 1]
    rewards = [summary_data[algo]['mean_reward'] for algo in algorithms]
    bars = ax.bar(algorithms, rewards, color=[colors[a] for a in algorithms])
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Mean Reward Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Path Efficiency
    ax = axes[1, 0]
    efficiency = [summary_data[algo]['mean_path_efficiency'] for algo in algorithms]
    bars = ax.bar(algorithms, efficiency, color=[colors[a] for a in algorithms])
    ax.set_ylabel('Path Efficiency', fontsize=12, fontweight='bold')
    ax.set_title('Path Efficiency Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Collision Rate
    ax = axes[1, 1]
    collision_rates = [summary_data[algo]['collision_rate'] * 100 for algo in algorithms]
    bars = ax.bar(algorithms, collision_rates, color=[colors[a] for a in algorithms])
    ax.set_ylabel('Collision Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Collision Rate Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'performance_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_difficulty_analysis(detailed_data, output_dir):
    """Анализ по уровням сложности"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance by Difficulty Level', 
                 fontsize=16, fontweight='bold')
    
    algorithms = list(detailed_data.keys())
    colors = {'ppo': '#3498db', 'sac': '#e74c3c', 'td3': '#2ecc71', 'dqn': '#f39c12'}
    
    # Success rate по сложности
    ax = axes[0]
    
    difficulty_bins = [(0, 0.33, 'Easy'), (0.33, 0.66, 'Medium'), (0.66, 1.0, 'Hard')]
    
    x = np.arange(len(difficulty_bins))
    width = 0.2
    
    for i, algo in enumerate(algorithms):
        results = detailed_data[algo]
        
        success_by_diff = []
        for min_d, max_d, label in difficulty_bins:
            filtered = [r for r in results if min_d <= r.get('difficulty', 0.5) < max_d]
            if filtered:
                success_rate = sum(r['success'] for r in filtered) / len(filtered) * 100
            else:
                success_rate = 0
            success_by_diff.append(success_rate)
        
        offset = (i - len(algorithms)/2 + 0.5) * width
        ax.bar(x + offset, success_by_diff, width, label=algo.upper(),
              color=colors[algo])
    
    ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate by Difficulty', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in difficulty_bins])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Performance degradation
    ax = axes[1]
    
    degradations = []
    for algo in algorithms:
        results = detailed_data[algo]
        
        easy = [r for r in results if r.get('difficulty', 0.5) < 0.33]
        hard = [r for r in results if r.get('difficulty', 0.5) >= 0.66]
        
        easy_success = sum(r['success'] for r in easy) / len(easy) * 100 if easy else 0
        hard_success = sum(r['success'] for r in hard) / len(hard) * 100 if hard else 0
        
        degradation = easy_success - hard_success
        degradations.append(degradation)
    
    bars = ax.barh(algorithms, degradations, color=[colors[a] for a in algorithms])
    ax.set_xlabel('Performance Degradation (Easy → Hard) %', fontsize=12, fontweight='bold')
    ax.set_title('Robustness to Difficulty', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, degradations)):
        ax.text(val, i, f' {val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'difficulty_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    # Paths
    results_file = Path('results/test_suite_evaluation_1m/20251024_023742/summary.json')
    output_dir = Path('results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print("📊 Loading results...")
    summary_data, detailed_data = load_results(results_file)
    
    print("📊 Generating performance summary...")
    plot_performance_summary(summary_data, output_dir)
    
    print("📊 Generating difficulty analysis...")
    plot_difficulty_analysis(detailed_data, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved in {output_dir}")
    
    # Печатаем summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (1M Timesteps)")
    print("="*60)
    for algo in summary_data.keys():
        data = summary_data[algo]
        print(f"\n{algo.upper()}:")
        print(f"  Success Rate:     {data['success_rate']*100:.1f}%")
        print(f"  Mean Reward:      {data['mean_reward']:.2f}")
        print(f"  Path Efficiency:  {data['mean_path_efficiency']:.3f}")
        print(f"  Collision Rate:   {data['collision_rate']*100:.1f}%")
        print(f"  Episode Length:   {data['mean_episode_length']:.1f}")


if __name__ == "__main__":
    main()

