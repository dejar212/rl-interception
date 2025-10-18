"""
Параллельное обучение всех алгоритмов на одинаковых условиях.
"""
import argparse
import sys
from pathlib import Path
import subprocess
from datetime import datetime
import json
import time

sys.path.append(str(Path(__file__).parent.parent))

from utils import load_config


def train_algorithm_subprocess(
    algo_name: str,
    env_config_path: str,
    algo_config_path: str,
    seed: int,
    output_dir: Path,
):
    """Запустить обучение алгоритма в отдельном процессе."""
    
    algo_output_dir = output_dir / algo_name.lower()
    
    cmd = [
        "python3",
        "experiments/train.py",
        "--env-config", env_config_path,
        "--algo-config", algo_config_path,
        "--seed", str(seed),
        "--output-dir", str(algo_output_dir),
    ]
    
    print(f"Starting {algo_name} training...")
    print(f"Command: {' '.join(cmd)}")
    
    # Запускаем процесс
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    
    return process


def main():
    parser = argparse.ArgumentParser(description='Parallel training of all algorithms')
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['ppo', 'sac', 'td3', 'dqn'],
        choices=['ppo', 'sac', 'td3', 'dqn'],
        help='Algorithms to train'
    )
    parser.add_argument(
        '--env-config',
        type=str,
        default='configs/env_default.yaml',
        help='Environment config'
    )
    parser.add_argument(
        '--config-prefix',
        type=str,
        default='balanced',
        help='Prefix for algorithm configs (e.g., "balanced" -> "balanced_ppo.yaml")'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (same for all algorithms)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/parallel_training',
        help='Output directory'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run sequentially instead of parallel (for debugging)'
    )
    
    args = parser.parse_args()
    
    # Создать output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PARALLEL TRAINING EXPERIMENT")
    print("="*70)
    print(f"Algorithms: {', '.join([a.upper() for a in args.algorithms])}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print("="*70)
    print()
    
    # Сохраняем конфигурацию эксперимента
    experiment_config = {
        'algorithms': args.algorithms,
        'env_config': args.env_config,
        'config_prefix': args.config_prefix,
        'seed': args.seed,
        'timestamp': timestamp,
        'mode': 'sequential' if args.sequential else 'parallel',
    }
    
    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    processes = {}
    start_times = {}
    
    if args.sequential:
        # Последовательное обучение
        for algo in args.algorithms:
            algo_config_path = f"configs/{args.config_prefix}_{algo}.yaml"
            
            print(f"\n{'='*70}")
            print(f"Training {algo.upper()}")
            print(f"{'='*70}\n")
            
            start_time = time.time()
            process = train_algorithm_subprocess(
                algo,
                args.env_config,
                algo_config_path,
                args.seed,
                output_dir,
            )
            
            # Ждем завершения
            stdout, stderr = process.communicate()
            
            elapsed = time.time() - start_time
            
            if process.returncode == 0:
                print(f"\n✅ {algo.upper()} completed successfully in {elapsed:.1f}s")
            else:
                print(f"\n❌ {algo.upper()} failed!")
                print(f"Error: {stderr}")
    
    else:
        # Параллельное обучение
        for algo in args.algorithms:
            algo_config_path = f"configs/{args.config_prefix}_{algo}.yaml"
            
            process = train_algorithm_subprocess(
                algo,
                args.env_config,
                algo_config_path,
                args.seed,
                output_dir,
            )
            
            processes[algo] = process
            start_times[algo] = time.time()
        
        print(f"\n{'='*70}")
        print("All processes started. Waiting for completion...")
        print(f"{'='*70}\n")
        
        # Мониторим процессы
        completed = set()
        while len(completed) < len(processes):
            for algo, process in processes.items():
                if algo in completed:
                    continue
                
                # Проверяем завершился ли процесс
                retcode = process.poll()
                if retcode is not None:
                    elapsed = time.time() - start_times[algo]
                    completed.add(algo)
                    
                    if retcode == 0:
                        print(f"✅ {algo.upper()} completed in {elapsed:.1f}s")
                    else:
                        print(f"❌ {algo.upper()} failed (exit code: {retcode})")
            
            # Немного спим, чтобы не грузить CPU
            if len(completed) < len(processes):
                time.sleep(1)
        
        print(f"\n{'='*70}")
        print("All training processes completed!")
        print(f"{'='*70}\n")
    
    # Собираем пути к моделям
    models_info = {}
    for algo in args.algorithms:
        algo_dir = output_dir / algo.lower()
        
        # Ищем директорию с моделью (может быть с timestamp)
        model_dirs = list(algo_dir.glob("*"))
        if model_dirs:
            model_dir = model_dirs[0]  # Берем первую (должна быть только одна)
            model_path = model_dir / "models" / f"{algo}_model.pth"
            
            if model_path.exists():
                models_info[algo] = {
                    'model_path': str(model_path),
                    'algo_config': f"configs/{args.config_prefix}_{algo}.yaml",
                }
                print(f"{algo.upper()}: {model_path}")
            else:
                print(f"⚠️ {algo.upper()}: Model not found at {model_path}")
        else:
            print(f"⚠️ {algo.upper()}: No output directory found")
    
    # Сохраняем информацию о моделях
    models_info_path = output_dir / "trained_models.json"
    with open(models_info_path, 'w') as f:
        json.dump(models_info, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Parallel training experiment completed!")
    print(f"Results saved in: {output_dir}")
    print(f"Models info: {models_info_path}")
    print(f"{'='*70}\n")
    
    # Печатаем команду для следующего шага
    print("Next step - Test on test suite:")
    print(f"python3 experiments/test_on_suite.py \\")
    print(f"  --models-info {models_info_path} \\")
    print(f"  --test-suite configs/test_suite.json \\")
    print(f"  --output-dir results/test_suite_evaluation")


if __name__ == "__main__":
    main()

