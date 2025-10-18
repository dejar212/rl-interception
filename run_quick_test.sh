#!/bin/bash

# Быстрый тест эксперимента (меньше timesteps для проверки)
# Usage: ./run_quick_test.sh

set -e  # Exit on error

echo "=================================="
echo "🧪 Quick Test Experiment (Fast)"
echo "=================================="
echo ""

# Параметры для быстрого теста
SEED=42
ALGORITHMS="ppo sac"  # Только 2 алгоритма для скорости
CONFIG_PREFIX="balanced"
TEST_SUITE="configs/test_suite.json"

# Создаем временные конфигурации с меньшим количеством timesteps
echo "Creating quick test configs..."
mkdir -p configs/quick_test

for algo in ppo sac; do
    # Копируем и модифицируем конфиг
    cp configs/balanced_${algo}.yaml configs/quick_test/quick_${algo}.yaml
    
    # Заменяем total_timesteps на 10000 для быстрого теста
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's/total_timesteps: 200000/total_timesteps: 10000/' configs/quick_test/quick_${algo}.yaml
    else
        # Linux
        sed -i 's/total_timesteps: 200000/total_timesteps: 10000/' configs/quick_test/quick_${algo}.yaml
    fi
done

echo "  ✓ Quick configs created"
echo ""

# Шаг 1: Проверка тестовой выборки
echo "Step 1: Checking test suite..."
if [ ! -f "$TEST_SUITE" ]; then
    echo "  Generating test suite..."
    python3 utils/test_suite_generator.py
else
    echo "  ✓ Test suite already exists"
fi
echo ""

# Шаг 2: Быстрое обучение (только PPO и SAC, 10K timesteps)
echo "Step 2: Quick training (10K timesteps, 2 algorithms)..."
echo "  This should take ~5-10 minutes"
echo ""

python3 experiments/parallel_train.py \
    --algorithms $ALGORITHMS \
    --config-prefix quick_test/quick \
    --seed $SEED \
    --output-dir results/quick_test

# Находим директорию с результатами
TRAINING_DIR=$(ls -td results/quick_test/* | head -1)
MODELS_INFO="$TRAINING_DIR/trained_models.json"

echo ""
echo "  ✓ Training completed"
echo "  Results: $TRAINING_DIR"
echo ""

# Шаг 3: Тестирование только на 20 средах (для скорости)
echo "Step 3: Quick testing (20 environments)..."
echo ""

# Создаем маленькую тестовую выборку
python3 -c "
import json
with open('$TEST_SUITE', 'r') as f:
    data = json.load(f)
# Берем первые 20 сред
data['environments'] = data['environments'][:20]
data['metadata']['n_environments'] = 20
with open('configs/quick_test_suite.json', 'w') as f:
    json.dump(data, f, indent=2)
print('Created quick test suite with 20 environments')
"

python3 experiments/test_on_suite.py \
    --models-info "$MODELS_INFO" \
    --test-suite "configs/quick_test_suite.json" \
    --output-dir results/quick_test_evaluation \
    --deterministic

# Находим директорию с результатами тестирования
EVAL_DIR=$(ls -td results/quick_test_evaluation/* | head -1)

echo ""
echo "  ✓ Testing completed"
echo "  Results: $EVAL_DIR"
echo ""

# Финальная сводка
echo "=================================="
echo "✅ Quick test completed!"
echo "=================================="
echo ""
echo "Results locations:"
echo "  Training: $TRAINING_DIR"
echo "  Testing:  $EVAL_DIR"
echo ""
echo "Summary:"
cat "$EVAL_DIR/summary.json"
echo ""
echo "To run full experiment (200K timesteps, 4 algorithms, 100 envs):"
echo "  ./run_full_experiment.sh"
echo ""

