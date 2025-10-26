#!/usr/bin/env python3
"""
Создание анимаций лучших и худших случаев для каждого алгоритма
"""
import sys
sys.path.append('.')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, List
import argparse

from environment import InterceptionEnv
from algorithms import PPOAgent, SACAgent, TD3Agent, DQNAgent, A2CAgent, DDPGAgent
from utils import set_seed


class BestWorstAnimator:
    """Создатель анимаций лучших и худших исходов"""
    
    def __init__(self, models_info: Dict, output_dir: Path):
        self.models_info = models_info
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_classes = {
            'ppo': PPOAgent,
            'sac': SACAgent,
            'td3': TD3Agent,
            'dqn': DQNAgent,
            'a2c': A2CAgent,
            'ddpg': DDPGAgent
        }
    
    def load_agent(self, algo_name: str, model_path: str, config_path: str):
        """Загрузка агента"""
        import yaml
        
        # Загрузка конфигурации
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Создание среды
        env = InterceptionEnv()
        
        # Создание агента
        agent_class = self.agent_classes[algo_name]
        agent = agent_class(env, config['algorithm'])
        
        # Загрузка модели
        agent.load(model_path)
        
        return agent, env
    
    def find_best_worst_scenarios(self, algo_name: str, n_scenarios: int = 10):
        """Поиск лучших и худших сценариев для алгоритма"""
        print(f"\n🔍 Finding best and worst scenarios for {algo_name.upper()}...")
        
        model_path = self.models_info[algo_name]['model_path']
        config_path = self.models_info[algo_name]['algo_config']
        
        agent, env = self.load_agent(algo_name, model_path, config_path)
        
        scenarios = []
        
        for episode in range(n_scenarios):
            set_seed(42 + episode)
            obs, _ = env.reset()
            
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 500:
                action = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            
            success = info.get('success', False)
            collision = info.get('collision', False)
            
            scenarios.append({
                'seed': 42 + episode,
                'reward': episode_reward,
                'success': success,
                'collision': collision,
                'steps': steps
            })
        
        # Сортируем по награде
        scenarios.sort(key=lambda x: x['reward'], reverse=True)
        
        best = scenarios[0]
        worst = scenarios[-1]
        
        print(f"  ✅ Best: Reward={best['reward']:.2f}, Success={best['success']}, Steps={best['steps']}")
        print(f"  ❌ Worst: Reward={worst['reward']:.2f}, Success={worst['success']}, Steps={worst['steps']}")
        
        return best, worst
    
    def create_animation(self, algo_name: str, seed: int, title: str, output_file: Path):
        """Создание анимации для конкретного сценария"""
        print(f"🎬 Creating animation: {output_file.name}")
        
        model_path = self.models_info[algo_name]['model_path']
        config_path = self.models_info[algo_name]['algo_config']
        
        agent, env = self.load_agent(algo_name, model_path, config_path)
        
        # Запуск эпизода
        set_seed(seed)
        obs, _ = env.reset()
        
        states = []
        actions_taken = []
        rewards = []
        
        done = False
        steps = 0
        
        while not done and steps < 500:
            states.append({
                'agent_pos': env.agent_pos.copy(),
                'target_pos': env.target_pos.copy(),
                'obstacles': [(obs[0], obs[1], obs[2]) for obs in env.obstacles]
            })
            
            action = agent.predict(obs, deterministic=True)
            actions_taken.append(action.copy())
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            steps += 1
        
        # Финальное состояние
        states.append({
            'agent_pos': env.agent_pos.copy(),
            'target_pos': env.target_pos.copy(),
            'obstacles': [(obs[0], obs[1], obs[2]) for obs in env.obstacles]
        })
        
        # Создание анимации
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title}\nAlgorithm: {algo_name.upper()}', 
                     fontsize=14, fontweight='bold')
        
        # Настройка основного графика
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.3)
        
        # Рисуем препятствия
        for obs_x, obs_y, obs_r in states[0]['obstacles']:
            circle = plt.Circle((obs_x, obs_y), obs_r, color='gray', alpha=0.3)
            ax1.add_patch(circle)
        
        # Траектория агента
        agent_traj, = ax1.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Agent Path')
        # Траектория цели
        target_traj, = ax1.plot([], [], 'r--', alpha=0.5, linewidth=1, label='Target Path')
        # Текущие позиции
        agent_point, = ax1.plot([], [], 'bo', markersize=10, label='Agent')
        target_point, = ax1.plot([], [], 'r*', markersize=15, label='Target')
        
        ax1.legend(loc='upper right')
        
        # График награды
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Reward Progress')
        ax2.grid(True, alpha=0.3)
        reward_line, = ax2.plot([], [], 'g-', linewidth=2)
        
        def init():
            agent_traj.set_data([], [])
            target_traj.set_data([], [])
            agent_point.set_data([], [])
            target_point.set_data([], [])
            reward_line.set_data([], [])
            return agent_traj, target_traj, agent_point, target_point, reward_line
        
        def animate(frame):
            if frame >= len(states):
                frame = len(states) - 1
            
            # Траектории до текущего момента
            agent_x = [s['agent_pos'][0] for s in states[:frame+1]]
            agent_y = [s['agent_pos'][1] for s in states[:frame+1]]
            target_x = [s['target_pos'][0] for s in states[:frame+1]]
            target_y = [s['target_pos'][1] for s in states[:frame+1]]
            
            agent_traj.set_data(agent_x, agent_y)
            target_traj.set_data(target_x, target_y)
            
            # Текущие позиции
            agent_point.set_data([agent_x[-1]], [agent_y[-1]])
            target_point.set_data([target_x[-1]], [target_y[-1]])
            
            # Награда
            if frame < len(rewards):
                cum_rewards = np.cumsum(rewards[:frame+1])
                reward_line.set_data(range(len(cum_rewards)), cum_rewards)
                ax2.set_xlim(0, len(rewards))
                ax2.set_ylim(min(cum_rewards.min(), 0), max(cum_rewards.max(), 10))
            
            return agent_traj, target_traj, agent_point, target_point, reward_line
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(states), interval=50, blit=True)
        
        # Сохранение
        anim.save(str(output_file), writer='pillow', fps=20)
        plt.close()
        
        print(f"  ✅ Saved animation: {output_file}")
    
    def create_all_animations(self):
        """Создание анимаций для всех алгоритмов"""
        for algo_name in self.models_info.keys():
            print(f"\n{'='*60}")
            print(f"Processing {algo_name.upper()}")
            print(f"{'='*60}")
            
            try:
                # Находим лучший и худший сценарии
                best, worst = self.find_best_worst_scenarios(algo_name, n_scenarios=20)
                
                # Создаем анимации
                best_file = self.output_dir / f'{algo_name}_best_case.gif'
                self.create_animation(algo_name, best['seed'], 
                                    f'Best Case - Reward: {best["reward"]:.2f}',
                                    best_file)
                
                worst_file = self.output_dir / f'{algo_name}_worst_case.gif'
                self.create_animation(algo_name, worst['seed'],
                                    f'Worst Case - Reward: {worst["reward"]:.2f}',
                                    worst_file)
                
            except Exception as e:
                print(f"❌ Error processing {algo_name}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Create best/worst case animations")
    parser.add_argument('--models-info', type=str,
                       default='results/parallel_training_1m/20251024_001927/trained_models.json',
                       help='Path to trained_models.json')
    parser.add_argument('--output-dir', type=str,
                       default='results/animations',
                       help='Output directory for animations')
    args = parser.parse_args()
    
    # Загрузка информации о моделях
    models_info_file = Path(args.models_info)
    if not models_info_file.exists():
        print(f"❌ Models info not found: {models_info_file}")
        print("Creating demo structure...")
        
        # Демо структура
        models_info = {
            'ppo': {
                'model_path': 'results/parallel_training_1m/20251024_001927/ppo/ppo_*/models/ppo_model.pth',
                'algo_config': 'configs/long_ppo.yaml'
            },
            'sac': {
                'model_path': 'results/parallel_training_1m/20251024_001927/sac/ppo_*/models/sac_model.pth',
                'algo_config': 'configs/long_sac.yaml'
            }
        }
    else:
        with open(models_info_file, 'r') as f:
            models_info = json.load(f)
    
    # Создаем выходную директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем аниматор
    animator = BestWorstAnimator(models_info, output_dir)
    
    print("\n🎬 Starting animation creation...")
    print(f"Output directory: {output_dir}")
    
    animator.create_all_animations()
    
    print(f"\n✅ All animations created! Check {output_dir}")


if __name__ == "__main__":
    main()

