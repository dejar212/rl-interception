"""
Скрипт для обучения RL агента на задаче перехвата.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from environment import InterceptionEnv
from algorithms.ppo import PPOAgent
from utils import set_seed, load_config, merge_configs
from visualization import plot_trajectory, plot_training_metrics


def train_ppo(
    env: InterceptionEnv,
    agent: PPOAgent,
    total_timesteps: int,
    num_steps: int,
    batch_size: int,
    num_epochs: int,
    save_dir: Path,
):
    """
    Обучение PPO агента.
    
    Args:
        env: Среда для обучения
        agent: PPO агент
        total_timesteps: Общее количество шагов обучения
        num_steps: Шагов на rollout
        batch_size: Размер батча
        num_epochs: Эпох обучения на rollout
        save_dir: Директория для сохранения результатов
    """
    # Метрики
    episode_rewards = []
    episode_lengths = []
    success_flags = []
    
    # Буферы для rollout
    obs_buffer = []
    actions_buffer = []
    log_probs_buffer = []
    rewards_buffer = []
    values_buffer = []
    dones_buffer = []
    
    # Инициализация
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    global_step = 0
    num_updates = 0
    
    print("Starting training...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Steps per rollout: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs per update: {num_epochs}\n")
    
    while global_step < total_timesteps:
        # Collect rollout
        for step in range(num_steps):
            global_step += 1
            episode_length += 1
            
            # Получаем действие от агента
            action = agent.predict(obs, deterministic=False)
            
            # Получаем log_prob и value
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                _, log_prob, _, value = agent.actor_critic.get_action_and_value(
                    obs_tensor,
                    torch.FloatTensor(action).unsqueeze(0).to(agent.device)
                )
                log_prob = log_prob.cpu().numpy()[0]
                value = value.cpu().numpy()[0][0]
            
            # Выполняем шаг в среде
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Сохраняем в буферы
            obs_buffer.append(obs)
            actions_buffer.append(action)
            log_probs_buffer.append(log_prob)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            dones_buffer.append(done)
            
            episode_reward += reward
            obs = next_obs
            
            # Конец эпизода
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                success_flags.append(info.get('success', False))
                
                # Логируем каждые 10 эпизодов
                if len(episode_rewards) % 10 == 0:
                    recent_rewards = episode_rewards[-10:]
                    recent_success = success_flags[-10:]
                    print(f"Episode {len(episode_rewards)} | "
                          f"Steps: {global_step} | "
                          f"Reward: {np.mean(recent_rewards):.2f} | "
                          f"Success: {np.mean(recent_success):.2%} | "
                          f"Length: {episode_length}")
                
                # Сбрасываем среду
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            
            if global_step >= total_timesteps:
                break
        
        # Compute next value для GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            next_value = agent.actor_critic.get_value(obs_tensor).cpu().numpy()[0][0]
        
        # Вычисляем advantages и returns
        advantages, returns = agent.compute_gae(
            np.array(rewards_buffer),
            np.array(values_buffer),
            np.array(dones_buffer),
            next_value
        )
        
        # Нормализация advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Конвертируем в тензоры
        obs_tensor = torch.FloatTensor(np.array(obs_buffer)).to(agent.device)
        actions_tensor = torch.FloatTensor(np.array(actions_buffer)).to(agent.device)
        log_probs_tensor = torch.FloatTensor(np.array(log_probs_buffer)).to(agent.device)
        returns_tensor = torch.FloatTensor(returns).to(agent.device)
        advantages_tensor = torch.FloatTensor(advantages).to(agent.device)
        
        # Обучаем агента
        for epoch in range(num_epochs):
            # Mini-batch training
            indices = np.arange(len(obs_buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(obs_buffer), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                metrics = agent.train_step(
                    obs_tensor[batch_indices],
                    actions_tensor[batch_indices],
                    log_probs_tensor[batch_indices],
                    returns_tensor[batch_indices],
                    advantages_tensor[batch_indices],
                )
        
        num_updates += 1
        
        # Очищаем буферы
        obs_buffer.clear()
        actions_buffer.clear()
        log_probs_buffer.clear()
        rewards_buffer.clear()
        values_buffer.clear()
        dones_buffer.clear()
    
    print("\nTraining completed!")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Success rate (last 100): {np.mean(success_flags[-100:]):.2%}")
    
    return episode_rewards, episode_lengths, success_flags


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for interception task')
    parser.add_argument(
        '--env-config',
        type=str,
        default='configs/env_default.yaml',
        help='Path to environment config'
    )
    parser.add_argument(
        '--algo-config',
        type=str,
        default='configs/ppo_config.yaml',
        help='Path to algorithm config'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Загружаем конфигурации
    print("Loading configurations...")
    env_config = load_config(args.env_config)
    algo_config = load_config(args.algo_config)
    
    # Устанавливаем seed
    seed = args.seed if args.seed is not None else env_config['environment'].get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Создаем директорию для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / f"ppo_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # Создаем среду
    print("\nCreating environment...")
    env = InterceptionEnv(**env_config['environment'])
    
    # Создаем агента
    print("Creating PPO agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    algo_params = algo_config['algorithm']
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=algo_params.get('hidden_dim', 64),
        learning_rate=algo_params.get('learning_rate', 3e-4),
        gamma=algo_params.get('gamma', 0.99),
        gae_lambda=algo_params.get('gae_lambda', 0.95),
        clip_coef=algo_params.get('clip_coef', 0.2),
        vf_coef=algo_params.get('vf_coef', 0.5),
        ent_coef=algo_params.get('ent_coef', 0.01),
        max_grad_norm=algo_params.get('max_grad_norm', 0.5),
        device=algo_params.get('device', 'cpu'),
    )
    
    # Обучаем агента
    print("\n" + "="*60)
    episode_rewards, episode_lengths, success_flags = train_ppo(
        env=env,
        agent=agent,
        total_timesteps=algo_params.get('total_timesteps', 100000),
        num_steps=algo_params.get('num_steps', 2048),
        batch_size=algo_params.get('batch_size', 64),
        num_epochs=algo_params.get('num_epochs', 10),
        save_dir=save_dir,
    )
    print("="*60)
    
    # Сохраняем модель
    model_path = save_dir / "models" / "ppo_model.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Визуализация метрик обучения
    print("\nGenerating training plots...")
    success_rate = []
    window = 20
    for i in range(len(success_flags)):
        start_idx = max(0, i - window + 1)
        success_rate.append(np.mean(success_flags[start_idx:i+1]))
    
    plot_training_metrics(
        rewards=episode_rewards,
        success_rate=success_rate,
        episode_lengths=episode_lengths,
        save_path=save_dir / "plots" / "training_metrics.png",
        show=False,
    )
    
    # Тестовый эпизод с визуализацией траектории
    print("\nRunning test episode for trajectory visualization...")
    obs, _ = env.reset(seed=seed)
    done = False
    
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    agent_traj, target_traj = env.get_trajectories()
    obstacles = info['obstacles']
    
    plot_trajectory(
        agent_trajectory=agent_traj,
        target_trajectory=target_traj,
        obstacles=obstacles,
        area_size=env_config['environment']['area_size'],
        save_path=save_dir / "plots" / "test_trajectory.png",
        show=False,
    )
    
    print("\n✅ Training completed successfully!")
    print(f"All results saved in: {save_dir}")


if __name__ == "__main__":
    main()

