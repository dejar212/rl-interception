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
from algorithms import PPOAgent, SACAgent, TD3Agent, DQNAgent
from utils import set_seed, load_config, merge_configs
from visualization import plot_trajectory, plot_training_metrics


def create_agent(algo_name: str, obs_dim: int, action_dim: int, algo_params: dict):
    """Create agent based on algorithm name"""
    if algo_name.lower() == 'ppo':
        return PPOAgent(
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
    elif algo_name.lower() == 'sac':
        return SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_params.get('hidden_dim', 64),
            learning_rate=algo_params.get('learning_rate', 3e-4),
            gamma=algo_params.get('gamma', 0.99),
            tau=algo_params.get('tau', 0.005),
            alpha=algo_params.get('alpha', 0.2),
            automatic_entropy_tuning=algo_params.get('automatic_entropy_tuning', True),
            buffer_size=algo_params.get('buffer_size', 100000),
            device=algo_params.get('device', 'cpu'),
        )
    elif algo_name.lower() == 'td3':
        return TD3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_params.get('hidden_dim', 64),
            learning_rate=algo_params.get('learning_rate', 3e-4),
            gamma=algo_params.get('gamma', 0.99),
            tau=algo_params.get('tau', 0.005),
            policy_noise=algo_params.get('policy_noise', 0.2),
            noise_clip=algo_params.get('noise_clip', 0.5),
            policy_delay=algo_params.get('policy_delay', 2),
            exploration_noise=algo_params.get('exploration_noise', 0.1),
            buffer_size=algo_params.get('buffer_size', 100000),
            max_action=algo_params.get('max_action', 1.0),
            device=algo_params.get('device', 'cpu'),
        )
    elif algo_name.lower() == 'dqn':
        return DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_params.get('hidden_dim', 64),
            learning_rate=algo_params.get('learning_rate', 3e-4),
            gamma=algo_params.get('gamma', 0.99),
            epsilon_start=algo_params.get('epsilon_start', 1.0),
            epsilon_end=algo_params.get('epsilon_end', 0.01),
            epsilon_decay=algo_params.get('epsilon_decay', 0.995),
            tau=algo_params.get('tau', 1.0),
            target_update_frequency=algo_params.get('target_update_frequency', 1000),
            buffer_size=algo_params.get('buffer_size', 100000),
            n_discrete_actions=algo_params.get('n_discrete_actions', 5),
            device=algo_params.get('device', 'cpu'),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def train_agent(
    env: InterceptionEnv,
    agent,
    total_timesteps: int,
    save_dir: Path,
    algo_name: str,
):
    """Universal training function for all algorithms"""
    if algo_name.lower() == 'ppo':
        # Get PPO-specific parameters from config
        from utils import load_config
        import sys
        # Find algo config from sys.argv
        algo_config_path = None
        for i, arg in enumerate(sys.argv):
            if arg == '--algo-config' and i + 1 < len(sys.argv):
                algo_config_path = sys.argv[i + 1]
                break
        
        # Load config and extract PPO parameters
        num_steps = 2048
        batch_size = 64
        num_epochs = 10
        if algo_config_path:
            try:
                config = load_config(algo_config_path)
                algo_params = config.get('algorithm', {})
                num_steps = algo_params.get('num_steps', 2048)
                batch_size = algo_params.get('batch_size', 64)
                num_epochs = algo_params.get('num_epochs', 10)
            except:
                pass  # Use defaults if config loading fails
        
        return train_ppo_universal(env, agent, total_timesteps, save_dir, 
                                   num_steps, batch_size, num_epochs)
    else:
        return train_off_policy(env, agent, total_timesteps, save_dir, algo_name)


def train_ppo_universal(
    env: InterceptionEnv,
    agent,
    total_timesteps: int,
    save_dir: Path,
    num_steps: int = 2048,
    batch_size: int = 64,
    num_epochs: int = 10,
):
    """PPO training function with configurable parameters"""
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
    obs_tuple = env.reset()
    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    episode_reward = 0
    episode_length = 0
    global_step = 0
    
    print(f"Training PPO for {total_timesteps} timesteps...")
    
    while global_step < total_timesteps:
        # Collect rollout (num_steps from config)
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
            next_obs_tuple, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs_tuple[0] if isinstance(next_obs_tuple, tuple) else next_obs_tuple
            done = terminated or truncated
            
            # Сохраняем в буферы
            obs_buffer.append(obs)
            actions_buffer.append(action)
            log_probs_buffer.append(log_prob)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            dones_buffer.append(done)
            
            obs = next_obs
            episode_reward += reward
            
            # Конец эпизода
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                success_flags.append(info.get('success', False))
                
                # Прогресс
                if len(episode_rewards) % 10 == 0:
                    recent_rewards = episode_rewards[-10:]
                    recent_success = sum(success_flags[-10:]) / len(success_flags[-10:])
                    print(f"Episode {len(episode_rewards)}, "
                          f"Reward: {np.mean(recent_rewards):.2f}, "
                          f"Success: {recent_success:.2%}, "
                          f"Timestep: {global_step}")
                
                obs_tuple = env.reset()
                obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
                episode_reward = 0
                episode_length = 0
        
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
        
        # Обучаем агента (batch_size and num_epochs from config)
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
        
        # Очищаем буферы
        obs_buffer.clear()
        actions_buffer.clear()
        log_probs_buffer.clear()
        rewards_buffer.clear()
        values_buffer.clear()
        dones_buffer.clear()
    
    print(f"Training completed!")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Success rate (last 100): {np.mean(success_flags[-100:]):.2%}")
    
    return episode_rewards, episode_lengths, success_flags


def train_off_policy(
    env: InterceptionEnv,
    agent,
    total_timesteps: int,
    save_dir: Path,
    algo_name: str,
):
    """Training function for off-policy algorithms (SAC, TD3, DQN)"""
    episode_rewards = []
    episode_lengths = []
    success_flags = []
    
    obs_tuple = env.reset()
    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    episode_reward = 0
    episode_length = 0
    
    timestep = 0
    
    print(f"Training {algo_name.upper()} for {total_timesteps} timesteps...")
    
    while timestep < total_timesteps:
        # Выбираем действие
        action = agent.predict(obs)
        
        # Выполняем действие
        next_obs_tuple, reward, done, truncated, info = env.step(action)
        next_obs = next_obs_tuple[0] if isinstance(next_obs_tuple, tuple) else next_obs_tuple
        
        # Сохраняем опыт
        agent.store_transition(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        timestep += 1
        
        # Обучение
        learning_starts = 1000  # Default value
        train_frequency = 1    # Default value
        gradient_steps = 1      # Default value
        batch_size = 256       # Default value
        
        if timestep > learning_starts and timestep % train_frequency == 0:
            for _ in range(gradient_steps):
                agent.train_step(batch_size)
        
        # Конец эпизода
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success_flags.append(info.get('success', False))
            
            # Прогресс
            if len(episode_rewards) % 10 == 0:
                recent_rewards = episode_rewards[-10:]
                recent_success = sum(success_flags[-10:]) / len(success_flags[-10:])
                print(f"Episode {len(episode_rewards)}, "
                      f"Reward: {np.mean(recent_rewards):.2f}, "
                      f"Success: {recent_success:.2%}, "
                      f"Timestep: {timestep}")
            
            obs_tuple = env.reset()
            obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
            episode_reward = 0
            episode_length = 0
    
    print(f"Training completed!")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Success rate (last 100): {np.mean(success_flags[-100:]):.2%}")
    
    return episode_rewards, episode_lengths, success_flags


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
        '--algo',
        type=str,
        default='ppo',
        choices=['ppo', 'sac', 'td3', 'dqn'],
        help='Algorithm to use'
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
    print(f"Creating {args.algo.upper()} agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    algo_params = algo_config['algorithm']
    agent = create_agent(args.algo, obs_dim, action_dim, algo_params)
    
    # Обучаем агента
    print("\n" + "="*60)
    episode_rewards, episode_lengths, success_flags = train_agent(
        env=env,
        agent=agent,
        total_timesteps=algo_params.get('total_timesteps', 100000),
        save_dir=save_dir,
        algo_name=args.algo,
    )
    print("="*60)
    
    # Сохраняем модель
    model_path = save_dir / "models" / f"{args.algo}_model.pth"
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

