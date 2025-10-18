# RL-Interception

Проект для анализа и сравнения RL алгоритмов на задаче перехвата цели в среде с препятствиями.

## Описание

RL-агент должен минимизировать время перехвата движущейся цели в непрерывной 2D среде с круглыми препятствиями. Проект позволяет обучать и сравнивать различные алгоритмы обучения с подкреплением (PPO, SAC, TD3, DQN).

## Установка

### Требования
- Python 3.10 или выше
- pip

### Установка зависимостей

```bash
# Клонировать репозиторий
cd rl-interception

# Создать виртуальное окружение (рекомендуется)
python3 -m venv venv
source venv/bin/activate  # для Linux/Mac
# или
venv\Scripts\activate  # для Windows

# Установить зависимости
pip install -r requirements.txt
```

## Структура проекта

```
rl-interception/
├── configs/           # YAML конфигурации экспериментов
├── algorithms/        # RL алгоритмы (PPO, SAC, TD3, DQN)
├── environment/       # Среда перехвата с препятствиями
├── visualization/     # Визуализация и анализ
├── experiments/       # Скрипты для запуска экспериментов
├── results/          # Результаты экспериментов (не в git)
├── utils/            # Утилиты
└── tests/            # Тесты
```

## Быстрый старт

### Полный эксперимент (рекомендуется)

```bash
# Запустить полный эксперимент сравнения всех алгоритмов
./run_full_experiment.sh

# Или вручную:
# 1. Обучить все алгоритмы параллельно
python3 experiments/parallel_train.py --algorithms ppo sac td3 dqn

# 2. Протестировать на 100 средах
python3 experiments/test_on_suite.py \
  --models-info results/parallel_training/*/trained_models.json \
  --test-suite configs/test_suite.json
```

См. [EXPERIMENT_GUIDE.md](./EXPERIMENT_GUIDE.md) для подробностей.

### Обучение одного агента

```bash
# PPO
python3 experiments/train.py --algo-config configs/balanced_ppo.yaml

# Для других алгоритмов
python3 experiments/train.py --algo-config configs/balanced_sac.yaml
```

### Оценка обученной модели

```bash
python3 experiments/evaluate.py \
  --model results/ppo_*/models/ppo_model.pth \
  --agent-type ppo \
  --algo-config configs/balanced_ppo.yaml \
  --n-episodes 100 \
  --deterministic
```

### Сравнение алгоритмов

```bash
python3 experiments/compare.py \
  --algorithms ppo sac td3 \
  --timesteps 50000
```

## Документация

- [idea.md](./idea.md) - Описание идеи проекта
- [problem.md](./problem.md) - Формальная постановка задачи
- [vision.md](./vision.md) - Техническое видение и архитектура
- [tasklist.md](./tasklist.md) - План разработки

## Версии

- **v0.1** ✅ - Базовый прототип с PPO
- **v0.2** ✅ - SAC, TD3, DQN алгоритмы + инструменты сравнения и анимация
- **v0.3** ✅ - Батчевое тестирование, расширенные метрики, тепловые карты, автоматическая генерация отчетов

## Лицензия

MIT

