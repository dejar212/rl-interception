#!/bin/bash

# Полный эксперимент по сравнению RL алгоритмов
# Usage: ./run_full_experiment.sh

set -e  # Exit on error

echo "=================================="
echo "🔬 Full Algorithm Comparison Experiment"
echo "=================================="
echo ""

# Параметры
SEED=42
ALGORITHMS="ppo sac td3 dqn"
CONFIG_PREFIX="balanced"
TEST_SUITE="configs/test_suite.json"

# Шаг 1: Проверка тестовой выборки
echo "Step 1: Checking test suite..."
if [ ! -f "$TEST_SUITE" ]; then
    echo "  Generating test suite..."
    python3 utils/test_suite_generator.py
else
    echo "  ✓ Test suite already exists"
fi
echo ""

# Шаг 2: Параллельное обучение
echo "Step 2: Training all algorithms..."
echo "  This will take some time (~30-60 minutes depending on hardware)"
echo ""

python3 experiments/parallel_train.py \
    --algorithms $ALGORITHMS \
    --config-prefix $CONFIG_PREFIX \
    --seed $SEED \
    --output-dir results/parallel_training

# Находим директорию с результатами (последняя созданная)
TRAINING_DIR=$(ls -td results/parallel_training/* | head -1)
MODELS_INFO="$TRAINING_DIR/trained_models.json"

echo ""
echo "  ✓ Training completed"
echo "  Results: $TRAINING_DIR"
echo ""

# Шаг 3: Тестирование на выборке
echo "Step 3: Testing on test suite (100 environments)..."
echo ""

python3 experiments/test_on_suite.py \
    --models-info "$MODELS_INFO" \
    --test-suite "$TEST_SUITE" \
    --output-dir results/test_suite_evaluation \
    --deterministic

# Находим директорию с результатами тестирования
EVAL_DIR=$(ls -td results/test_suite_evaluation/* | head -1)

echo ""
echo "  ✓ Testing completed"
echo "  Results: $EVAL_DIR"
echo ""

# Финальная сводка
echo "=================================="
echo "✅ Experiment completed successfully!"
echo "=================================="
echo ""
echo "Results locations:"
echo "  Training: $TRAINING_DIR"
echo "  Testing:  $EVAL_DIR"
echo ""
echo "Check these files:"
echo "  - $EVAL_DIR/summary.json"
echo "  - $EVAL_DIR/plots/summary_table.png"
echo "  - $EVAL_DIR/plots/success_by_difficulty.png"
echo ""
echo "For detailed analysis, see:"
echo "  - $EVAL_DIR/detailed_results.json"
echo ""

