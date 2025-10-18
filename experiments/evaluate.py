"""
Скрипт для оценки обученной модели на задаче перехвата.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from environment import InterceptionEnv
from algorithms import PPOAgent, SACAgent, TD3Agent, DQNAgent
from utils import set_seed, load_config
from visualization import plot_trajectory


def load_agent(agent_type: str, model_path: str, obs_dim: int, action_dim: int, config: dict, device: str):
    """Загрузить обученного агента."""
    if agent_type.upper() == 'PPO':
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.get('hidden_dim', 64),
            device=device,
        )
    elif agent_type.upper() == 'SAC':
        agent = SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.get('hidden_dim', 256),
            device=device,
        )
    elif agent_type.upper() == 'TD3':
        agent = TD3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.get('hidden_dim', 256),
            device=device,
        )
    elif agent_type.upper() == 'DQN':
        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.get('hidden_dim', 128),
            n_discrete_actions=config.get('n_discrete_actions', 5),
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load(model_path)
    return agent


def evaluate_agent(
    env: InterceptionEnv,
    agent,
    n_episodes: int = 100,
    deterministic: bool = True,
    visualize: bool = False,
    save_dir: Path = None,
):
    """
    Оценить агента на среде.
    
    Args:
        env: Среда для оценки
        agent: Обученный агент
        n_episodes: Количество эпизодов для оценки
        deterministic: Детерминистическая политика
        visualize: Визуализировать траектории
        save_dir: Директория для сохранения результатов
    
    Returns:
        Словарь с метриками
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    total_collisions = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_had_collision = False
        
        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('collision', False):
                episode_had_collision = True
                total_collisions += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get('success', False):
            success_count += 1
        
        if episode_had_collision:
            collision_count += 1
        
        # Визуализация первых нескольких эпизодов
        if visualize and episode < 5 and save_dir is not None:
            agent_traj, target_traj = env.get_trajectories()
            obstacles = info['obstacles']
            
            plot_trajectory(
                agent_trajectory=agent_traj,
                target_trajectory=target_traj,
                obstacles=obstacles,
                area_size=env.area_size,
                save_path=save_dir / f"trajectory_ep{episode+1}.png",
                show=False,
            )
        
        # Прогресс
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Success rate: {success_count/(episode+1):.2%}, "
                  f"Avg reward: {np.mean(episode_rewards[-10:]):.2f}")
    
    # Вычисляем метрики
    metrics = {
        'n_episodes': n_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'success_rate': float(success_count / n_episodes),
        'collision_rate': float(collision_count / n_episodes),
        'avg_collisions_per_episode': float(total_collisions / n_episodes),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--agent-type',
        type=str,
        required=True,
        choices=['ppo', 'sac', 'td3', 'dqn'],
        help='Type of agent'
    )
    parser.add_argument(
        '--env-config',
        type=str,
        default='configs/env_default.yaml',
        help='Path to environment config'
    )
    parser.add_argument(
        '--algo-config',
        type=str,
        default=None,
        help='Path to algorithm config (optional)'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic policy'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save trajectory visualizations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Установить seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Загрузить конфигурации
    print("Loading configurations...")
    env_config = load_config(args.env_config)
    
    algo_config = {}
    if args.algo_config:
        algo_config = load_config(args.algo_config).get('algorithm', {})
    
    # Создать среду
    print("Creating environment...")
    env = InterceptionEnv(**env_config['environment'])
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = algo_config.get('device', 'cpu')
    
    # Загрузить агента
    print(f"Loading {args.agent_type.upper()} agent from {args.model}...")
    agent = load_agent(
        agent_type=args.agent_type,
        model_path=args.model,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=algo_config,
        device=device,
    )
    
    # Создать директорию для результатов
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Оценить агента
    print("\n" + "="*60)
    print(f"Evaluating {args.agent_type.upper()} agent...")
    print(f"Episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("="*60 + "\n")
    
    metrics = evaluate_agent(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        visualize=args.visualize,
        save_dir=save_dir if args.visualize else None,
    )
    
    # Вывести результаты
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Episodes: {metrics['n_episodes']}")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Collision Rate: {metrics['collision_rate']:.2%}")
    print(f"Avg Collisions/Episode: {metrics['avg_collisions_per_episode']:.2f}")
    print(f"Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print("="*60)
    
    # Сохранить метрики
    metrics_path = save_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print(f"\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

