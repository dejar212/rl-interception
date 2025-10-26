# RL-Interception

Comprehensive framework for analyzing and comparing RL algorithms on target interception with obstacles.

## 🌟 Highlights

- ✅ **6 RL Algorithms**: PPO, SAC, TD3, DQN, A2C, DDPG
- ✅ **1M Timesteps Training**: Extended training for optimal performance
- ✅ **100 Test Environments**: Comprehensive evaluation suite
- ✅ **Advanced Analytics**: Learning curves, convergence analysis, applicability studies
- ✅ **Visual Demonstrations**: Animations of best/worst cases
- ✅ **Production Ready**: SAC achieves 28% success rate

## Описание

RL-агент должен минимизировать время перехвата движущейся цели в непрерывной 2D среде с круглыми препятствиями. Проект позволяет обучать и сравнивать различные алгоритмы обучения с подкреплением (PPO, SAC, TD3, DQN, A2C, DDPG).

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
- **v0.4** ✅ - **NEW!** A2C, DDPG алгоритмы + анализ темпов обучения + сравнение по условиям

## 🆕 Новые возможности (v0.4)

### Дополнительные алгоритмы
- **A2C** (Advantage Actor-Critic) - on-policy алгоритм с параллельными акторами
- **DDPG** (Deep Deterministic Policy Gradient) - детерминистический off-policy алгоритм

### Расширенная аналитика

#### 1. Анализ темпов обучения
```bash
python3 experiments/analyze_learning.py --output-dir results/analysis
```
- Learning curves для всех алгоритмов
- Анализ скорости сходимости
- Сравнение достижения асимптоты
- Автоматический отчет

#### 2. Анализ по условиям среды
```bash
python3 experiments/compare_by_conditions.py --output-dir results/analysis
```
- Производительность vs количество препятствий
- Анализ применимости по сложности
- Деградация производительности
- Рекомендации по использованию

#### 3. Анимации лучших/худших случаев
```bash
python3 experiments/create_best_worst_animations.py --output-dir results/animations
```
- Автоматический поиск лучших и худших сценариев
- Создание GIF анимаций
- Визуализация траекторий
- Графики наград в реальном времени

### Комплексный анализ

См. **[COMPREHENSIVE_ANALYSIS.md](./COMPREHENSIVE_ANALYSIS.md)** для полного отчета:
- 📊 Детальные бенчмарки (1M timesteps)
- 🚀 Анализ динамики обучения
- 🎯 Применимость по условиям
- 💡 Рекомендации по выбору алгоритма
- 🏆 SAC - лучший алгоритм (28% success rate)

## 🎯 Ключевые результаты

| Метрика | PPO | SAC 🏆 | TD3 | DQN | A2C | DDPG |
|---------|-----|--------|-----|-----|-----|------|
| Success Rate | 2% | **28%** | 13% | 4% | TBD | TBD |
| Path Efficiency | 0.636 | **0.750** | 0.681 | 0.728 | TBD | TBD |
| Episode Length | 587.8 | **474.1** | 532.2 | 576.9 | TBD | TBD |
| Collision Rate | 32% | 52% | 50% | 70% | TBD | TBD |

**Вывод**: SAC показывает наилучшую производительность на задаче перехвата с препятствиями.

## Лицензия

MIT

