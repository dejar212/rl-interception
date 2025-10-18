"""
Батчевое тестирование алгоритмов на нескольких сценариях.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

from environment import InterceptionEnv
from algorithms import PPOAgent, SACAgent, TD3Agent, DQNAgent
from utils import set_seed, load_config, MetricsCalculator, ReportGenerator
from visualization import (
    plot_heatmap,
    plot_box_comparison,
    plot_violin_comparison,
    plot_metric_grid,
    plot_performance_radar,
)


def create_agent(agent_type: str, obs_dim: int, action_dim: int, model_path: str, config: dict):
    """Создать и загрузить агента."""
    algo_config = config.get('algorithm', {})
    
    if agent_type.upper() == 'PPO':
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('hidden_dim', 64),
            device=algo_config.get('device', 'cpu'),
        )
    elif agent_type.upper() == 'SAC':
        agent = SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('hidden_dim', 256),
            device=algo_config.get('device', 'cpu'),
        )
    elif agent_type.upper() == 'TD3':
        agent = TD3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('hidden_dim', 256),
            device=algo_config.get('device', 'cpu'),
        )
    elif agent_type.upper() == 'DQN':
        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('hidden_dim', 128),
            n_discrete_actions=algo_config.get('n_discrete_actions', 5),
            device=algo_config.get('device', 'cpu'),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load(model_path)
    return agent


def test_on_scenario(
    env: InterceptionEnv,
    agent,
    n_episodes: int,
    deterministic: bool = True,
) -> MetricsCalculator:
    """
    Тестировать агента на одном сценарии.
    
    Returns:
        MetricsCalculator с результатами
    """
    metrics = MetricsCalculator()
    
    for episode in range(n_episodes):
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
        
        # Собираем данные
        success = info.get('success', False)
        agent_traj, target_traj = env.get_trajectories()
        
        interception_time = episode_length if success else None
        
        metrics.add_episode(
            reward=episode_reward,
            length=episode_length,
            success=success,
            collision=had_collision,
            collision_count=collision_count,
            interception_time=interception_time,
            agent_trajectory=agent_traj,
            target_start=target_traj[0] if len(target_traj) > 0 else None,
            target_end=target_traj[-1] if len(target_traj) > 0 else None,
        )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Batch testing on multiple scenarios')
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Paths to trained models'
    )
    parser.add_argument(
        '--agent-types',
        nargs='+',
        required=True,
        choices=['ppo', 'sac', 'td3', 'dqn'],
        help='Types of agents (must match --models order)'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        default=['configs/scenario_easy.yaml',
                 'configs/scenario_hard.yaml',
                 'configs/scenario_extreme.yaml'],
        help='Scenario config files'
    )
    parser.add_argument(
        '--algo-configs',
        nargs='+',
        help='Algorithm config files (must match --models order)'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100,
        help='Number of test episodes per scenario'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/batch_test',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Проверка соответствия
    if len(args.models) != len(args.agent_types):
        print("Error: Number of models must match number of agent types")
        return
    
    if args.algo_configs and len(args.algo_configs) != len(args.models):
        print("Error: Number of algo configs must match number of models")
        return
    
    # Создать директорию результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("BATCH TESTING")
    print("="*70)
    print(f"Models: {len(args.models)}")
    print(f"Scenarios: {len(args.scenarios)}")
    print(f"Episodes per scenario: {args.n_episodes}")
    print(f"Output: {output_dir}")
    print("="*70)
    print()
    
    # Установить seed
    set_seed(args.seed)
    
    # Результаты: {algorithm: {scenario: metrics}}
    all_results = {}
    
    # Тестируем каждый алгоритм на каждом сценарии
    for model_idx, (model_path, agent_type) in enumerate(zip(args.models, args.agent_types)):
        algo_name = agent_type.upper()
        all_results[algo_name] = {}
        
        print(f"\n{'='*70}")
        print(f"Testing {algo_name}")
        print(f"Model: {model_path}")
        print(f"{'='*70}\n")
        
        # Загрузить конфигурацию алгоритма
        algo_config = {}
        if args.algo_configs:
            algo_config = load_config(args.algo_configs[model_idx])
        
        for scenario_path in args.scenarios:
            scenario_name = Path(scenario_path).stem
            print(f"  Scenario: {scenario_name}...")
            
            # Загрузить сценарий
            scenario_config = load_config(scenario_path)
            
            # Создать среду
            env = InterceptionEnv(**scenario_config['environment'])
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            # Загрузить агента
            agent = create_agent(agent_type, obs_dim, action_dim, model_path, algo_config)
            
            # Тестировать
            metrics = test_on_scenario(env, agent, args.n_episodes, deterministic=True)
            all_results[algo_name][scenario_name] = metrics
            
            # Краткие результаты
            summary = metrics.get_summary()
            print(f"    Success rate: {summary['success_rate']:.2%}")
            print(f"    Mean reward: {summary['mean_reward']:.2f}")
            print(f"    Collision rate: {summary['collision_rate']:.2%}")
    
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS AND REPORTS")
    print(f"{'='*70}\n")
    
    # Создаем сравнительные визуализации для каждого сценария
    for scenario_path in args.scenarios:
        scenario_name = Path(scenario_path).stem
        print(f"  Generating plots for {scenario_name}...")
        
        # Собираем данные по сценарию
        scenario_data = {}
        for algo_name in all_results.keys():
            metrics = all_results[algo_name][scenario_name]
            scenario_data[algo_name] = metrics
        
        # Box plots для разных метрик
        rewards_data = {
            algo: metrics.episode_rewards
            for algo, metrics in scenario_data.items()
        }
        plot_box_comparison(
            rewards_data,
            metric_name="Reward",
            save_path=plots_dir / f"{scenario_name}_rewards_box.png",
            show=False,
        )
        
        # Violin plots для success
        success_data = {
            algo: [1.0 if s else 0.0 for s in metrics.success_flags]
            for algo, metrics in scenario_data.items()
        }
        plot_violin_comparison(
            success_data,
            metric_name="Success",
            save_path=plots_dir / f"{scenario_name}_success_violin.png",
            show=False,
        )
        
        # Тепловые карты для каждого алгоритма
        for algo_name, metrics in scenario_data.items():
            positions = metrics.get_positions_array()
            if len(positions) > 0:
                plot_heatmap(
                    positions,
                    save_path=plots_dir / f"{scenario_name}_{algo_name.lower()}_heatmap.png",
                    show=False,
                    title=f"{algo_name} - {scenario_name}"
                )
    
    # Сводные метрики по всем сценариям
    print("  Generating summary visualizations...")
    
    # Усреднение метрик по сценариям
    avg_metrics = {}
    for algo_name in all_results.keys():
        scenario_summaries = [
            all_results[algo_name][Path(s).stem].get_summary()
            for s in args.scenarios
        ]
        
        avg_metrics[algo_name] = {
            'mean_reward': np.mean([s['mean_reward'] for s in scenario_summaries]),
            'success_rate': np.mean([s['success_rate'] for s in scenario_summaries]),
            'collision_rate': np.mean([s['collision_rate'] for s in scenario_summaries]),
            'mean_episode_length': np.mean([s['mean_episode_length'] for s in scenario_summaries]),
            'mean_path_efficiency': np.mean([
                s.get('mean_path_efficiency', 0) for s in scenario_summaries
            ]),
        }
    
    # Metric grid
    plot_metric_grid(
        avg_metrics,
        save_path=plots_dir / "summary_metrics_grid.png",
        show=False,
    )
    
    # Radar chart
    plot_performance_radar(
        avg_metrics,
        save_path=plots_dir / "summary_radar.png",
        show=False,
    )
    
    # Сохраняем детальные результаты
    print("  Saving detailed results...")
    
    detailed_results = {}
    for algo_name in all_results.keys():
        detailed_results[algo_name] = {}
        for scenario_path in args.scenarios:
            scenario_name = Path(scenario_path).stem
            metrics = all_results[algo_name][scenario_name]
            detailed_results[algo_name][scenario_name] = metrics.get_summary()
            
            # Сохраняем CSV
            csv_path = output_dir / f"{algo_name}_{scenario_name}_episodes.csv"
            metrics.save_to_csv(str(csv_path))
    
    # Сохраняем JSON
    json_path = output_dir / "summary.json"
    with open(json_path, 'w') as f:
        json.dump({
            'algorithms': list(all_results.keys()),
            'scenarios': [Path(s).stem for s in args.scenarios],
            'results': detailed_results,
            'averaged_metrics': avg_metrics,
        }, f, indent=2)
    
    # Генерируем отчет
    print("  Generating report...")
    report_gen = ReportGenerator(output_dir)
    report_path = report_gen.generate_comparison_report(
        algorithms=list(all_results.keys()),
        metrics_dict=avg_metrics,
        experiment_config={
            'scenarios': args.scenarios,
            'n_episodes_per_scenario': args.n_episodes,
            'seed': args.seed,
        },
        plots_dir=plots_dir,
    )
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nAverage metrics across all scenarios:\n")
    
    # Печатаем итоговую таблицу
    print(f"{'Algorithm':<12} {'Success Rate':<15} {'Mean Reward':<15} {'Collision Rate':<15}")
    print("-" * 60)
    for algo_name, metrics in avg_metrics.items():
        print(f"{algo_name:<12} "
              f"{metrics['success_rate']:<15.2%} "
              f"{metrics['mean_reward']:<15.2f} "
              f"{metrics['collision_rate']:<15.2%}")
    
    print(f"\n{'='*70}")
    print(f"✅ Batch testing completed!")
    print(f"Results saved in: {output_dir}")
    print(f"Report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

