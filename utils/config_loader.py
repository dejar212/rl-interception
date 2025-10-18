"""
Загрузка и валидация YAML конфигурационных файлов.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Загружает YAML конфигурацию из файла.
    
    Args:
        config_path: Путь к YAML файлу
    
    Returns:
        Словарь с конфигурацией
    
    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: Если файл содержит невалидный YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Объединяет несколько конфигураций.
    
    Конфигурации объединяются слева направо, 
    более поздние значения перезаписывают ранние.
    
    Args:
        *configs: Произвольное количество словарей конфигураций
    
    Returns:
        Объединенный словарь конфигурации
    """
    result = {}
    
    for config in configs:
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Рекурсивное глубокое объединение двух словарей.
    
    Args:
        dict1: Базовый словарь
        dict2: Словарь для объединения
    
    Returns:
        Объединенный словарь
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], output_path: Union[str, Path]):
    """
    Сохраняет конфигурацию в YAML файл.
    
    Args:
        config: Словарь конфигурации
        output_path: Путь для сохранения
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

