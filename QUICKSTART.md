# Quick Start Guide

Быстрое руководство по запуску проекта RL-Interception.

## Установка

```bash
# 1. Перейдите в директорию проекта
cd rl-interception

# 2. (Опционально) Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# 3. Установите зависимости
pip install -r requirements.txt
```

## Быстрый тест (5-10 минут)

Запустите короткое обучение для проверки работоспособности:

```bash
python3 experiments/train.py --algo-config configs/test_config.yaml --seed 42
```

Результаты сохранятся в `results/ppo_TIMESTAMP/`

## Полное обучение (~30-60 минут)

Запустите обучение с полными параметрами:

```bash
python3 experiments/train.py \
  --env-config configs/env_default.yaml \
  --algo-config configs/ppo_config.yaml \
  --seed 42
```

## Результаты

После обучения вы найдете:

```
results/ppo_TIMESTAMP/
├── models/
│   └── ppo_model.pth              # Обученная модель
└── plots/
    ├── training_metrics.png        # Графики обучения
    └── test_trajectory.png         # Траектория перехвата
```

## Параметры запуска

```bash
python3 experiments/train.py \
  --env-config PATH           # Конфигурация среды (default: configs/env_default.yaml)
  --algo-config PATH          # Конфигурация алгоритма (default: configs/ppo_config.yaml)
  --seed INT                  # Random seed (default: 42)
  --output-dir PATH           # Директория для результатов (default: results)
```

## Настройка параметров

### Изменить количество препятствий

Отредактируйте `configs/env_default.yaml`:

```yaml
environment:
  n_obstacles: 10  # Увеличить до 10 препятствий
```

### Изменить длительность обучения

Отредактируйте `configs/ppo_config.yaml`:

```yaml
algorithm:
  total_timesteps: 200000  # Удвоить время обучения
```

## Что дальше?

1. Экспериментируйте с параметрами среды
2. Попробуйте разные random seeds
3. Изменяйте reward функцию
4. Ждите версию 0.2 с дополнительными алгоритмами!

## Проблемы?

- **ImportError**: Проверьте установку зависимостей
- **CUDA errors**: Измените `device: "cpu"` в конфигурации алгоритма
- **Out of Memory**: Уменьшите `num_steps` или `batch_size` в конфигурации

---

Подробнее см. [README.md](./README.md) и [vision.md](./vision.md)

