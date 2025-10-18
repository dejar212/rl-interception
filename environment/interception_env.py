"""
Среда перехвата цели с препятствиями для Gymnasium.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from .obstacles import CircularObstacle, generate_obstacles, check_collision


class InterceptionEnv(gym.Env):
    """
    Среда для задачи перехвата движущейся цели с препятствиями.
    
    Observation Space:
        - agent_x, agent_y: позиция агента
        - agent_vx, agent_vy: скорость агента
        - target_x, target_y: текущая позиция цели
        - target_vx, target_vy: скорость цели
        - distance_to_target: расстояние до цели
        - obstacles: координаты и радиусы препятствий (опционально)
    
    Action Space:
        - Непрерывное управление: [ax, ay] - ускорение агента
    
    Reward:
        - Штраф за время (каждый шаг)
        - Штраф за столкновение с препятствиями
        - Награда за перехват цели
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        area_size: float = 1.0,
        n_obstacles: int = 5,
        obstacle_radius_range: Tuple[float, float] = (0.05, 0.15),
        max_episode_steps: int = 500,
        agent_max_speed: float = 0.02,
        agent_max_accel: float = 0.002,
        target_speed: float = 0.01,
        interception_radius: float = 0.03,
        agent_radius: float = 0.01,
        collision_penalty: float = -10.0,
        time_penalty: float = -0.01,
        success_reward: float = 100.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        # Параметры среды
        self.area_size = area_size
        self.n_obstacles = n_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.max_episode_steps = max_episode_steps
        self.agent_max_speed = agent_max_speed
        self.agent_max_accel = agent_max_accel
        self.target_speed = target_speed
        self.interception_radius = interception_radius
        self.agent_radius = agent_radius
        self.collision_penalty = collision_penalty
        self.time_penalty = time_penalty
        self.success_reward = success_reward
        
        # Размерность наблюдения
        # agent: x, y, vx, vy (4)
        # target: x, y, vx, vy (4)
        # distance to target (1)
        # obstacles: до 10 ближайших (x, y, radius) * 10 = 30
        # Всегда резервируем место под 10 препятствий для consistency
        obs_dim = 4 + 4 + 1 + (3 * 10)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: ускорение [ax, ay]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Состояние среды
        self.agent_pos = np.zeros(2)
        self.agent_vel = np.zeros(2)
        self.target_pos = np.zeros(2)
        self.target_vel = np.zeros(2)
        self.obstacles = []
        self.current_step = 0
        
        # Траектории для визуализации
        self.agent_trajectory = []
        self.target_trajectory = []
        
        self._seed = seed
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сброс среды в начальное состояние."""
        super().reset(seed=seed)
        
        if seed is not None:
            self._seed = seed
            self.np_random = np.random.RandomState(seed)
        
        # Генерируем препятствия
        self.obstacles = generate_obstacles(
            n_obstacles=self.n_obstacles,
            area_size=(self.area_size, self.area_size),
            radius_range=self.obstacle_radius_range,
            seed=self._seed
        )
        
        # Начальная позиция агента (случайная, но не в препятствии)
        max_attempts = 100
        for _ in range(max_attempts):
            self.agent_pos = self.np_random.uniform(0, self.area_size, size=2)
            if not check_collision(
                self.agent_pos[0],
                self.agent_pos[1],
                self.obstacles,
                self.agent_radius
            ):
                break
        
        self.agent_vel = np.zeros(2)
        
        # Начальная позиция и скорость цели
        self.target_pos = self.np_random.uniform(0, self.area_size, size=2)
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.target_vel = self.target_speed * np.array([np.cos(angle), np.sin(angle)])
        
        self.current_step = 0
        self.agent_trajectory = [self.agent_pos.copy()]
        self.target_trajectory = [self.target_pos.copy()]
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполнение одного шага среды."""
        self.current_step += 1
        
        # Применяем ускорение к агенту
        accel = np.clip(action, -1.0, 1.0) * self.agent_max_accel
        self.agent_vel += accel
        
        # Ограничиваем скорость агента
        speed = np.linalg.norm(self.agent_vel)
        if speed > self.agent_max_speed:
            self.agent_vel = self.agent_vel / speed * self.agent_max_speed
        
        # Обновляем позицию агента
        new_agent_pos = self.agent_pos + self.agent_vel
        
        # Проверяем столкновение с препятствиями
        collision = check_collision(
            new_agent_pos[0],
            new_agent_pos[1],
            self.obstacles,
            self.agent_radius
        )
        
        # Обновляем позицию (даже если столкновение)
        self.agent_pos = new_agent_pos
        
        # Обновляем позицию цели (прямолинейное движение)
        self.target_pos += self.target_vel
        
        # Сохраняем траектории
        self.agent_trajectory.append(self.agent_pos.copy())
        self.target_trajectory.append(self.target_pos.copy())
        
        # Вычисляем расстояние до цели
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        
        # Проверяем условия завершения
        success = distance <= self.interception_radius
        timeout = self.current_step >= self.max_episode_steps
        terminated = success
        truncated = timeout
        
        # Вычисляем награду
        reward = self.time_penalty
        
        if collision:
            reward += self.collision_penalty
        
        if success:
            reward += self.success_reward
        
        obs = self._get_observation()
        info = self._get_info()
        info['success'] = success
        info['collision'] = collision
        info['distance'] = distance
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Формирование вектора наблюдения."""
        # Базовые компоненты
        obs_parts = [
            self.agent_pos,
            self.agent_vel,
            self.target_pos,
            self.target_vel,
            [np.linalg.norm(self.agent_pos - self.target_pos)]
        ]
        
        # Добавляем информацию о ближайших препятствиях
        obstacle_obs = []
        for obs_obj in self.obstacles[:10]:  # Максимум 10 ближайших
            obstacle_obs.extend([obs_obj.x, obs_obj.y, obs_obj.radius])
        
        # Дополняем нулями если препятствий меньше 10
        while len(obstacle_obs) < 30:
            obstacle_obs.append(0.0)
        
        obs_parts.append(obstacle_obs)
        
        # Собираем в один массив
        observation = np.concatenate([
            np.array(part, dtype=np.float32).flatten() 
            for part in obs_parts
        ])
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Дополнительная информация о состоянии среды."""
        return {
            'agent_pos': self.agent_pos.copy(),
            'agent_vel': self.agent_vel.copy(),
            'target_pos': self.target_pos.copy(),
            'target_vel': self.target_vel.copy(),
            'step': self.current_step,
            'obstacles': [(o.x, o.y, o.radius) for o in self.obstacles],
        }
    
    def render(self):
        """Рендеринг среды (пока не реализован)."""
        pass
    
    def get_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Получить траектории агента и цели."""
        return (
            np.array(self.agent_trajectory),
            np.array(self.target_trajectory)
        )

