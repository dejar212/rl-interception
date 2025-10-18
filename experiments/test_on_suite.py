"""
Тестирование обученных моделей на тестовой выборке из 100 сред.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent))

from environment import InterceptionEnv
from algorithms import PPOAgent, SACAgent, TD3Agent, DQNAgent
from utils import set_seed, load_config
from visualization import (
    plot_box_comparison,
    plot_violin_comparison,
    plot_heatmap,
)
import matplotlib.pyplot as plt


def load_agent(
    agent_type: str,
    model_path: str,
    obs_dim: int,
    action_dim: int,
    algo_config: dict,
):
    """Загрузить обученного агента."""
    device = algo_config.get('algorithm', {}).get('device', 'cpu')
    
    if agent_type.upper() == 'PPO':
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('algorithm', {}).get('hidden_dim', 64),
            device=device,
        )
    elif agent_type.upper() == 'SAC':
        agent = SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('algorithm', {}).get('hidden_dim', 256),
            device=device,
        )
    elif agent_type.upper() == 'TD3':
        agent = TD3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('algorithm', {}).get('hidden_dim', 256),
            device=device,
        )
    elif agent_type.upper() == 'DQN':
        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('algorithm', {}).get('hidden_dim', 128),
            n_discrete_actions=algo_config.get('algorithm', {}).get('n_discrete_actions', 5),
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load(model_path)
    return agent


def test_agent_on_environment(
    agent,
    env_config: Dict[str, Any],
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Тестировать агента на одной среде.
    
    Returns:
        Словарь с результатами теста
    """
    env = InterceptionEnv(**env_config)
    
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    had_collision = False
    collision_count = 0
    
    while not done:
        action = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        
        if info.get('collision', False):
            had_collision = True
            collision_count += 1
    
    success = info.get('success', False)
    agent_traj, target_traj = env.get_trajectories()
    
    # Вычисляем path length
    if len(agent_traj) > 1:
        diffs = np.diff(agent_traj, axis=0)
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        path_length = 0.0
    
    # Прямое расстояние
    if len(target_traj) > 1:
        direct_distance = float(np.linalg.norm(target_traj[-1] - target_traj[0]))
        path_efficiency = direct_distance / path_length if path_length > 0 else 0.0
    else:
        direct_distance = 0.0
        path_efficiency = 0.0
    
    results = {
        'success': success,
        'reward': episode_reward,
        'length': episode_length,
        'collision': had_collision,
        'collision_count': collision_count,
        'path_length': path_length,
        'path_efficiency': min(path_efficiency, 1.0),
        'interception_time': episode_length if success else None,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test trained models on test suite')
    parser.add_argument(
        '--models-info',
        type=str,
        required=True,
        help='Path to trained_models.json from parallel_train.py'
    )
    parser.add_argument(
        '--test-suite',
        type=str,
        default='configs/test_suite.json',
        help='Path to test suite JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/test_suite_evaluation',
        help='Output directory'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='Use deterministic policy'
    )
    
    args = parser.parse_args()
    
    # Создать output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("TEST SUITE EVALUATION")
    print("="*70)
    print(f"Models info: {args.models_info}")
    print(f"Test suite: {args.test_suite}")
    print(f"Output: {output_dir}")
    print("="*70)
    print()
    
    # Загрузить информацию о моделях
    with open(args.models_info, 'r') as f:
        models_info = json.load(f)
    
    # Загрузить тестовую выборку
    with open(args.test_suite, 'r') as f:
        test_suite_data = json.load(f)
    
    test_environments = test_suite_data['environments']
    print(f"Loaded {len(test_environments)} test environments")
    print(f"Loaded {len(models_info)} trained models")
    print()
    
    # Результаты: {algorithm: [results_per_env]}
    all_results = {}
    
    # Тестируем каждый алгоритм
    for algo_name, model_data in models_info.items():
        print(f"\n{'='*70}")
        print(f"Testing {algo_name.upper()}")
        print(f"{'='*70}\n")
        
        model_path = model_data['model_path']
        algo_config_path = model_data['algo_config']
        
        # Загрузить конфигурацию
        algo_config = load_config(algo_config_path)
        
        # Создать пример среды для получения dimensions
        sample_env_config = {k: v for k, v in test_environments[0].items() 
                            if k not in ['id', 'seed', 'difficulty']}
        sample_env = InterceptionEnv(**sample_env_config)
        obs_dim = sample_env.observation_space.shape[0]
        action_dim = sample_env.action_space.shape[0]
        
        # Загрузить агента
        agent = load_agent(algo_name, model_path, obs_dim, action_dim, algo_config)
        
        # Тестировать на всех средах
        algo_results = []
        
        for i, env_config_data in enumerate(test_environments):
            # Подготовить конфигурацию среды
            env_config = {k: v for k, v in env_config_data.items() 
                         if k not in ['id', 'difficulty']}
            
            # Тестировать
            result = test_agent_on_environment(agent, env_config, args.deterministic)
            result['env_id'] = env_config_data['id']
            result['difficulty'] = env_config_data['difficulty']
            algo_results.append(result)
            
            # Прогресс
            if (i + 1) % 25 == 0:
                successes = sum(1 for r in algo_results if r['success'])
                print(f"  Progress: {i+1}/{len(test_environments)} - "
                      f"Success rate so far: {successes/(i+1):.2%}")
        
        all_results[algo_name] = algo_results
        
        # Печатаем итоговую статистику для алгоритма
        successes = sum(1 for r in algo_results if r['success'])
        rewards = [r['reward'] for r in algo_results]
        print(f"\n{algo_name.upper()} Results:")
        print(f"  Success rate: {successes/len(algo_results):.2%}")
        print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    # Сохранить детальные результаты
    print(f"\n{'='*70}")
    print("Saving results...")
    print(f"{'='*70}\n")
    
    results_path = output_dir / "detailed_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Detailed results saved to: {results_path}")
    
    # Агрегированные метрики
    aggregated = {}
    for algo_name, results in all_results.items():
        aggregated[algo_name] = {
            'success_rate': sum(1 for r in results if r['success']) / len(results),
            'mean_reward': float(np.mean([r['reward'] for r in results])),
            'std_reward': float(np.std([r['reward'] for r in results])),
            'mean_path_efficiency': float(np.mean([r['path_efficiency'] for r in results])),
            'collision_rate': sum(1 for r in results if r['collision']) / len(results),
            'mean_episode_length': float(np.mean([r['length'] for r in results])),
        }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    # Создать визуализации
    print(f"\nGenerating visualizations...")
    
    # 1. Box plots для разных метрик
    metrics_to_plot = [
        ('reward', 'Reward'),
        ('path_efficiency', 'Path Efficiency'),
        ('length', 'Episode Length'),
    ]
    
    for metric_key, metric_name in metrics_to_plot:
        data = {
            algo: [r[metric_key] for r in results]
            for algo, results in all_results.items()
        }
        
        plot_box_comparison(
            data,
            metric_name=metric_name,
            save_path=plots_dir / f"{metric_key}_box.png",
            show=False,
        )
        print(f"  ✓ {metric_name} box plot")
    
    # 2. Success rate по сложности
    difficulty_bins = [(0, 0.33, 'Easy'), (0.33, 0.66, 'Medium'), (0.66, 1.0, 'Hard')]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(difficulty_bins))
    width = 0.2
    
    for i, (algo_name, results) in enumerate(all_results.items()):
        success_rates = []
        
        for min_diff, max_diff, label in difficulty_bins:
            filtered = [r for r in results 
                       if min_diff <= r['difficulty'] < max_diff]
            if filtered:
                sr = sum(1 for r in filtered if r['success']) / len(filtered)
                success_rates.append(sr)
            else:
                success_rates.append(0)
        
        offset = (i - len(all_results)/2 + 0.5) * width
        ax.bar(x_pos + offset, success_rates, width, label=algo_name.upper())
    
    ax.set_xlabel('Difficulty Level', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate by Difficulty Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([label for _, _, label in difficulty_bins])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "success_by_difficulty.png", dpi=150)
    plt.close()
    print(f"  ✓ Success by difficulty plot")
    
    # 3. Сводная таблица
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Данные для таблицы
    table_data = [['Algorithm', 'Success Rate', 'Mean Reward', 'Path Efficiency', 'Collision Rate']]
    
    for algo_name, metrics in aggregated.items():
        row = [
            algo_name.upper(),
            f"{metrics['success_rate']:.2%}",
            f"{metrics['mean_reward']:.1f} ± {metrics['std_reward']:.1f}",
            f"{metrics['mean_path_efficiency']:.3f}",
            f"{metrics['collision_rate']:.2%}",
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Стиль заголовка
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Test Suite Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(plots_dir / "summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Summary table")
    
    # Финальная сводка
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Algorithm':<12} {'Success Rate':<15} {'Mean Reward':<20} {'Path Efficiency':<15}")
    print("-"*70)
    for algo_name, metrics in aggregated.items():
        print(f"{algo_name.upper():<12} "
              f"{metrics['success_rate']:<15.2%} "
              f"{metrics['mean_reward']:<8.1f} ± {metrics['std_reward']:<8.1f} "
              f"{metrics['mean_path_efficiency']:<15.3f}")
    
    print(f"\n{'='*70}")
    print(f"✅ Test suite evaluation completed!")
    print(f"Results: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

