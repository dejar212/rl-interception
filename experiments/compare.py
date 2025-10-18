"""
Скрипт для сравнения нескольких RL алгоритмов на задаче перехвата.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from environment import InterceptionEnv
from algorithms import PPOAgent, SACAgent, TD3Agent, DQNAgent
from utils import set_seed, load_config
from visualization import plot_training_metrics, plot_comparison
import matplotlib.pyplot as plt


def create_agent(agent_type: str, obs_dim: int, action_dim: int, config: dict):
    """Создать агента по типу."""
    algo_config = config.get('algorithm', {})
    
    if agent_type.upper() == 'PPO':
        return PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('hidden_dim', 64),
            learning_rate=algo_config.get('learning_rate', 3e-4),
            gamma=algo_config.get('gamma', 0.99),
            gae_lambda=algo_config.get('gae_lambda', 0.95),
            clip_coef=algo_config.get('clip_coef', 0.2),
            vf_coef=algo_config.get('vf_coef', 0.5),
            ent_coef=algo_config.get('ent_coef', 0.01),
            max_grad_norm=algo_config.get('max_grad_norm', 0.5),
            device=algo_config.get('device', 'cpu'),
        )
    elif agent_type.upper() == 'SAC':
        return SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('hidden_dim', 256),
            learning_rate=algo_config.get('learning_rate', 3e-4),
            gamma=algo_config.get('gamma', 0.99),
            tau=algo_config.get('tau', 0.005),
            alpha=algo_config.get('alpha', 0.2),
            automatic_entropy_tuning=algo_config.get('automatic_entropy_tuning', True),
            buffer_size=algo_config.get('buffer_size', 1000000),
            device=algo_config.get('device', 'cpu'),
        )
    elif agent_type.upper() == 'TD3':
        return TD3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('hidden_dim', 256),
            learning_rate=algo_config.get('learning_rate', 3e-4),
            gamma=algo_config.get('gamma', 0.99),
            tau=algo_config.get('tau', 0.005),
            policy_noise=algo_config.get('policy_noise', 0.2),
            noise_clip=algo_config.get('noise_clip', 0.5),
            policy_delay=algo_config.get('policy_delay', 2),
            exploration_noise=algo_config.get('exploration_noise', 0.1),
            buffer_size=algo_config.get('buffer_size', 1000000),
            max_action=algo_config.get('max_action', 1.0),
            device=algo_config.get('device', 'cpu'),
        )
    elif agent_type.upper() == 'DQN':
        return DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_config.get('hidden_dim', 128),
            learning_rate=algo_config.get('learning_rate', 1e-3),
            gamma=algo_config.get('gamma', 0.99),
            epsilon_start=algo_config.get('epsilon_start', 1.0),
            epsilon_end=algo_config.get('epsilon_end', 0.05),
            epsilon_decay=algo_config.get('epsilon_decay', 0.995),
            tau=algo_config.get('tau', 1.0),
            target_update_frequency=algo_config.get('target_update_frequency', 1000),
            buffer_size=algo_config.get('buffer_size', 100000),
            n_discrete_actions=algo_config.get('n_discrete_actions', 5),
            device=algo_config.get('device', 'cpu'),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_off_policy_agent(
    env: InterceptionEnv,
    agent,
    total_timesteps: int,
    learning_starts: int,
    batch_size: int,
    train_frequency: int,
    gradient_steps: int = 1,
):
    """Обучение off-policy агента (SAC, TD3, DQN)."""
    episode_rewards = []
    episode_lengths = []
    success_flags = []
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    print(f"Starting training (off-policy)...")
    
    for step in range(total_timesteps):
        # Получить действие
        action = agent.predict(obs, deterministic=False)
        
        # Выполнить шаг
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Сохранить в replay buffer
        agent.store_transition(obs, action, reward, next_obs, done)
        
        episode_reward += reward
        episode_length += 1
        obs = next_obs
        
        # Обучение
        if step >= learning_starts and step % train_frequency == 0:
            for _ in range(gradient_steps):
                agent.train_step(batch_size)
        
        # Конец эпизода
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success_flags.append(info.get('success', False))
            
            if len(episode_rewards) % 10 == 0:
                recent_rewards = episode_rewards[-10:]
                recent_success = success_flags[-10:]
                print(f"Episode {len(episode_rewards)} | "
                      f"Steps: {step+1} | "
                      f"Reward: {np.mean(recent_rewards):.2f} | "
                      f"Success: {np.mean(recent_success):.2%}")
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
    
    return episode_rewards, episode_lengths, success_flags


def main():
    parser = argparse.ArgumentParser(description='Compare multiple RL algorithms')
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['ppo', 'sac', 'td3'],
        choices=['ppo', 'sac', 'td3', 'dqn'],
        help='Algorithms to compare'
    )
    parser.add_argument(
        '--env-config',
        type=str,
        default='configs/env_default.yaml',
        help='Path to environment config'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='configs',
        help='Directory with algorithm configs'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Training timesteps per algorithm'
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
        default='results/comparison',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Создать директорию для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузить конфигурацию среды
    env_config = load_config(args.env_config)
    
    # Результаты
    all_results = {}
    
    # Обучить каждый алгоритм
    for algo_name in args.algorithms:
        print("\n" + "="*60)
        print(f"Training {algo_name.upper()}")
        print("="*60)
        
        # Установить seed
        set_seed(args.seed)
        
        # Загрузить конфигурацию алгоритма
        algo_config_path = Path(args.config_dir) / f"{algo_name}_config.yaml"
        algo_config = load_config(algo_config_path)
        
        # Создать среду
        env = InterceptionEnv(**env_config['environment'])
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Создать агента
        agent = create_agent(algo_name, obs_dim, action_dim, algo_config)
        
        # Обучить
        if algo_name.upper() == 'PPO':
            # PPO использует on-policy обучение (используем train.py логику)
            from experiments.train import train_ppo
            
            episode_rewards, episode_lengths, success_flags = train_ppo(
                env=env,
                agent=agent,
                total_timesteps=args.timesteps,
                num_steps=algo_config['algorithm'].get('num_steps', 2048),
                batch_size=algo_config['algorithm'].get('batch_size', 64),
                num_epochs=algo_config['algorithm'].get('num_epochs', 10),
                save_dir=save_dir,
            )
        else:
            # Off-policy алгоритмы (SAC, TD3, DQN)
            episode_rewards, episode_lengths, success_flags = train_off_policy_agent(
                env=env,
                agent=agent,
                total_timesteps=args.timesteps,
                learning_starts=algo_config['algorithm'].get('learning_starts', 1000),
                batch_size=algo_config['algorithm'].get('batch_size', 256),
                train_frequency=algo_config['algorithm'].get('train_frequency', 1),
                gradient_steps=algo_config['algorithm'].get('gradient_steps', 1),
            )
        
        # Сохранить результаты
        all_results[algo_name.upper()] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'success': success_flags,
        }
        
        # Сохранить модель
        model_path = save_dir / f"{algo_name}_model.pth"
        agent.save(str(model_path))
        print(f"Model saved to: {model_path}")
    
    # Сравнительная визуализация
    print("\n" + "="*60)
    print("Generating comparison plots...")
    print("="*60)
    
    # Выравниваем длины для сравнения (берем минимальную)
    min_episodes = min(len(results['rewards']) for results in all_results.values())
    
    rewards_comparison = {
        name: results['rewards'][:min_episodes]
        for name, results in all_results.items()
    }
    
    plot_comparison(
        results=rewards_comparison,
        metric='reward',
        save_path=save_dir / 'rewards_comparison.png',
        show=False,
    )
    
    # Success rate comparison
    success_comparison = {}
    window = 20
    for name, results in all_results.items():
        success_flags = results['success'][:min_episodes]
        success_rate = []
        for i in range(len(success_flags)):
            start_idx = max(0, i - window + 1)
            success_rate.append(np.mean(success_flags[start_idx:i+1]))
        success_comparison[name] = success_rate
    
    plot_comparison(
        results=success_comparison,
        metric='success rate',
        save_path=save_dir / 'success_comparison.png',
        show=False,
    )
    
    # Сохранить сводку
    summary = {}
    for name, results in all_results.items():
        summary[name] = {
            'mean_reward': float(np.mean(results['rewards'][-100:])),
            'success_rate': float(np.mean(results['success'][-100:])),
            'total_episodes': len(results['rewards']),
        }
    
    summary_path = save_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Вывести итоговую таблицу
    print("\n" + "="*60)
    print("Comparison Summary (last 100 episodes):")
    print("="*60)
    print(f"{'Algorithm':<12} {'Mean Reward':<15} {'Success Rate':<15} {'Episodes':<10}")
    print("-"*60)
    for name, metrics in summary.items():
        print(f"{name:<12} {metrics['mean_reward']:<15.2f} "
              f"{metrics['success_rate']:<15.2%} {metrics['total_episodes']:<10}")
    print("="*60)
    
    print(f"\n✅ Comparison completed!")
    print(f"Results saved in: {save_dir}")


if __name__ == "__main__":
    main()

