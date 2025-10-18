"""
Генерация и проверка препятствий для среды перехвата.
"""
import numpy as np
from typing import List, Tuple


class CircularObstacle:
    """Круглое препятствие в 2D пространстве."""
    
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius
    
    def contains_point(self, x: float, y: float) -> bool:
        """Проверяет, находится ли точка внутри препятствия."""
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius ** 2
    
    def intersects(self, other: 'CircularObstacle') -> bool:
        """Проверяет, пересекается ли с другим препятствием."""
        dist = np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        return dist < (self.radius + other.radius)


def generate_obstacles(
    n_obstacles: int,
    area_size: Tuple[float, float],
    radius_range: Tuple[float, float],
    seed: int = None,
    max_attempts: int = 1000
) -> List[CircularObstacle]:
    """
    Генерирует список круглых препятствий без сильного перекрытия.
    
    Args:
        n_obstacles: Количество препятствий
        area_size: Размер области (width, height)
        radius_range: Диапазон радиусов (min, max)
        seed: Random seed для воспроизводимости
        max_attempts: Максимум попыток размещения каждого препятствия
    
    Returns:
        Список объектов CircularObstacle
    """
    if seed is not None:
        np.random.seed(seed)
    
    obstacles = []
    width, height = area_size
    min_radius, max_radius = radius_range
    
    for _ in range(n_obstacles):
        placed = False
        
        for _ in range(max_attempts):
            # Генерируем случайную позицию и радиус
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            radius = np.random.uniform(min_radius, max_radius)
            
            new_obstacle = CircularObstacle(x, y, radius)
            
            # Проверяем, не пересекается ли сильно с существующими
            valid = True
            for existing in obstacles:
                if new_obstacle.intersects(existing):
                    valid = False
                    break
            
            if valid:
                obstacles.append(new_obstacle)
                placed = True
                break
        
        if not placed:
            # Если не удалось разместить, все равно добавляем хоть что-то
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            radius = np.random.uniform(min_radius, max_radius)
            obstacles.append(CircularObstacle(x, y, radius))
    
    return obstacles


def check_collision(
    x: float,
    y: float,
    obstacles: List[CircularObstacle],
    agent_radius: float = 0.0
) -> bool:
    """
    Проверяет столкновение точки с препятствиями.
    
    Args:
        x, y: Координаты точки
        obstacles: Список препятствий
        agent_radius: Радиус агента (для коллизии)
    
    Returns:
        True если есть столкновение, False иначе
    """
    for obstacle in obstacles:
        dist = np.sqrt((x - obstacle.x) ** 2 + (y - obstacle.y) ** 2)
        if dist <= (obstacle.radius + agent_radius):
            return True
    return False

