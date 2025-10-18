"""
Генератор тестовой выборки разнообразных сред для оценки.
"""
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any


class TestSuiteGenerator:
    """Генератор тестовых сценариев с разнообразными условиями."""
    
    def __init__(self, base_config: Dict[str, Any], seed: int = 42):
        """
        Args:
            base_config: Базовая конфигурация среды
            seed: Random seed для воспроизводимости
        """
        self.base_config = base_config
        self.seed = seed
        np.random.seed(seed)
    
    def generate_test_suite(self, n_environments: int = 100) -> List[Dict[str, Any]]:
        """
        Сгенерировать набор тестовых сред.
        
        Варьируемые параметры:
        - Количество препятствий
        - Размеры препятствий
        - Начальные позиции агента и цели
        - Скорость цели
        - Размер области
        
        Args:
            n_environments: Количество тестовых сред
        
        Returns:
            Список конфигураций сред
        """
        test_configs = []
        
        for i in range(n_environments):
            # Используем разные seeds для каждой среды
            env_seed = self.seed + i
            
            # Варьируем сложность среды
            difficulty = i / n_environments  # От 0 (легко) до 1 (сложно)
            
            config = self._generate_single_config(env_seed, difficulty, i)
            test_configs.append(config)
        
        return test_configs
    
    def _generate_single_config(
        self,
        env_seed: int,
        difficulty: float,
        index: int,
    ) -> Dict[str, Any]:
        """Сгенерировать одну конфигурацию среды."""
        
        # Базовые параметры
        config = {
            'id': index,
            'seed': env_seed,
            'difficulty': difficulty,
        }
        
        # Количество препятствий: от 2 до 15
        n_obstacles = int(2 + difficulty * 13)
        config['n_obstacles'] = n_obstacles
        
        # Размер препятствий: варьируем min и max радиус
        # Легкие среды - маленькие препятствия, сложные - большие
        min_radius = 0.03 + difficulty * 0.02  # 0.03 -> 0.05
        max_radius = 0.08 + difficulty * 0.12  # 0.08 -> 0.20
        config['obstacle_radius_range'] = [min_radius, max_radius]
        
        # Размер области: иногда варьируем
        if np.random.random() < 0.2:  # 20% сред с другим размером
            config['area_size'] = np.random.choice([0.8, 1.0, 1.2])
        else:
            config['area_size'] = 1.0
        
        # Скорость цели: варьируем от медленной до быстрой
        base_speed = 0.01
        config['target_speed'] = base_speed * (0.5 + difficulty * 1.5)  # 0.005 -> 0.020
        
        # Скорость агента: иногда варьируем
        if np.random.random() < 0.3:  # 30% сред
            config['agent_max_speed'] = 0.015 + np.random.uniform(0, 0.01)
        else:
            config['agent_max_speed'] = 0.02
        
        # Ускорение агента
        config['agent_max_accel'] = config['agent_max_speed'] / 10
        
        # Радиус агента: иногда варьируем
        if np.random.random() < 0.2:
            config['agent_radius'] = 0.008 + np.random.uniform(0, 0.006)
        else:
            config['agent_radius'] = 0.01
        
        # Радиус перехвата
        config['interception_radius'] = 0.03
        
        # Rewards: варьируем штрафы в зависимости от сложности
        config['collision_penalty'] = -10.0 * (1 + difficulty)  # -10 -> -20
        config['time_penalty'] = -0.01 * (1 + difficulty * 0.5)  # -0.01 -> -0.015
        config['success_reward'] = 100.0 * (1 + difficulty)  # 100 -> 200
        
        # Max steps
        config['max_episode_steps'] = int(500 * (1 + difficulty * 0.4))  # 500 -> 700
        
        return config
    
    def save_test_suite(self, test_configs: List[Dict[str, Any]], output_path: str):
        """Сохранить тестовую выборку в JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'n_environments': len(test_configs),
                'base_seed': self.seed,
                'generation_params': {
                    'obstacle_range': [2, 15],
                    'radius_range': [[0.03, 0.08], [0.05, 0.20]],
                    'target_speed_range': [0.005, 0.020],
                }
            },
            'environments': test_configs,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Test suite saved to: {output_path}")
        print(f"Generated {len(test_configs)} test environments")
    
    def load_test_suite(self, input_path: str) -> List[Dict[str, Any]]:
        """Загрузить тестовую выборку из JSON."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        return data['environments']
    
    def get_statistics(self, test_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Получить статистику по тестовой выборке."""
        n_obstacles = [c['n_obstacles'] for c in test_configs]
        target_speeds = [c['target_speed'] for c in test_configs]
        difficulties = [c['difficulty'] for c in test_configs]
        
        stats = {
            'total_environments': len(test_configs),
            'obstacles': {
                'mean': float(np.mean(n_obstacles)),
                'min': int(np.min(n_obstacles)),
                'max': int(np.max(n_obstacles)),
                'std': float(np.std(n_obstacles)),
            },
            'target_speed': {
                'mean': float(np.mean(target_speeds)),
                'min': float(np.min(target_speeds)),
                'max': float(np.max(target_speeds)),
                'std': float(np.std(target_speeds)),
            },
            'difficulty': {
                'mean': float(np.mean(difficulties)),
                'distribution': {
                    'easy (0-0.33)': sum(1 for d in difficulties if d < 0.33),
                    'medium (0.33-0.66)': sum(1 for d in difficulties if 0.33 <= d < 0.66),
                    'hard (0.66-1.0)': sum(1 for d in difficulties if d >= 0.66),
                }
            }
        }
        
        return stats


def generate_balanced_test_suite(
    n_environments: int = 100,
    output_path: str = "configs/test_suite.json",
    seed: int = 42,
):
    """
    Удобная функция для генерации тестовой выборки.
    
    Args:
        n_environments: Количество сред
        output_path: Путь для сохранения
        seed: Random seed
    """
    # Базовая конфигурация
    base_config = {
        'area_size': 1.0,
        'n_obstacles': 5,
        'obstacle_radius_range': [0.05, 0.15],
        'max_episode_steps': 500,
        'agent_max_speed': 0.02,
        'agent_max_accel': 0.002,
        'agent_radius': 0.01,
        'target_speed': 0.01,
        'interception_radius': 0.03,
        'collision_penalty': -10.0,
        'time_penalty': -0.01,
        'success_reward': 100.0,
    }
    
    generator = TestSuiteGenerator(base_config, seed=seed)
    test_configs = generator.generate_test_suite(n_environments)
    generator.save_test_suite(test_configs, output_path)
    
    # Печатаем статистику
    stats = generator.get_statistics(test_configs)
    print("\nTest Suite Statistics:")
    print(f"Total environments: {stats['total_environments']}")
    print(f"Obstacles: {stats['obstacles']['min']}-{stats['obstacles']['max']} "
          f"(mean: {stats['obstacles']['mean']:.1f})")
    print(f"Target speed: {stats['target_speed']['min']:.4f}-{stats['target_speed']['max']:.4f} "
          f"(mean: {stats['target_speed']['mean']:.4f})")
    print(f"Difficulty distribution:")
    for level, count in stats['difficulty']['distribution'].items():
        print(f"  {level}: {count} environments")
    
    return test_configs


if __name__ == "__main__":
    # Генерируем тестовую выборку
    generate_balanced_test_suite(
        n_environments=100,
        output_path="configs/test_suite.json",
        seed=42
    )

