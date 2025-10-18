"""
Автоматическая генерация отчетов в Markdown.
"""
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json


class ReportGenerator:
    """Генератор отчетов о результатах экспериментов."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comparison_report(
        self,
        algorithms: List[str],
        metrics_dict: Dict[str, Dict[str, Any]],
        experiment_config: Dict[str, Any] = None,
        plots_dir: Path = None,
    ) -> str:
        """
        Сгенерировать отчет о сравнении алгоритмов.
        
        Args:
            algorithms: Список названий алгоритмов
            metrics_dict: Словарь {algorithm: {metrics}}
            experiment_config: Конфигурация эксперимента
            plots_dir: Директория с графиками
        
        Returns:
            Путь к сгенерированному отчету
        """
        report_path = self.output_dir / "comparison_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("# Algorithm Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Конфигурация эксперимента
            if experiment_config:
                f.write("## Experiment Configuration\n\n")
                f.write("```yaml\n")
                f.write(self._dict_to_yaml(experiment_config))
                f.write("```\n\n")
                f.write("---\n\n")
            
            # Сводная таблица
            f.write("## Summary Metrics\n\n")
            f.write(self._generate_metrics_table(algorithms, metrics_dict))
            f.write("\n---\n\n")
            
            # Детальные метрики для каждого алгоритма
            f.write("## Detailed Metrics by Algorithm\n\n")
            for algo in algorithms:
                f.write(f"### {algo.upper()}\n\n")
                metrics = metrics_dict.get(algo, {})
                f.write(self._generate_detailed_metrics(metrics))
                f.write("\n")
            
            f.write("---\n\n")
            
            # Визуализации
            if plots_dir and plots_dir.exists():
                f.write("## Visualizations\n\n")
                f.write(self._generate_plots_section(plots_dir))
                f.write("\n---\n\n")
            
            # Выводы и рекомендации
            f.write("## Analysis and Recommendations\n\n")
            f.write(self._generate_recommendations(algorithms, metrics_dict))
            f.write("\n---\n\n")
            
            # Footer
            f.write("*Report generated automatically by RL-Interception framework*\n")
        
        return str(report_path)
    
    def _dict_to_yaml(self, d: Dict, indent: int = 0) -> str:
        """Конвертировать словарь в YAML строку."""
        lines = []
        prefix = "  " * indent
        
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._dict_to_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    def _generate_metrics_table(
        self,
        algorithms: List[str],
        metrics_dict: Dict[str, Dict[str, Any]],
    ) -> str:
        """Сгенерировать таблицу с метриками."""
        # Определяем ключевые метрики
        key_metrics = [
            ('mean_reward', 'Mean Reward'),
            ('success_rate', 'Success Rate'),
            ('collision_rate', 'Collision Rate'),
            ('mean_episode_length', 'Avg Episode Length'),
            ('mean_path_efficiency', 'Path Efficiency'),
        ]
        
        # Header
        table = "| Algorithm | " + " | ".join([name for _, name in key_metrics]) + " |\n"
        table += "|" + "---|" * (len(key_metrics) + 1) + "\n"
        
        # Rows
        for algo in algorithms:
            metrics = metrics_dict.get(algo, {})
            row = f"| **{algo.upper()}** |"
            
            for key, _ in key_metrics:
                value = metrics.get(key, 'N/A')
                if isinstance(value, float):
                    if 'rate' in key or 'efficiency' in key:
                        row += f" {value:.2%} |"
                    else:
                        row += f" {value:.2f} |"
                else:
                    row += f" {value} |"
            
            table += row + "\n"
        
        return table
    
    def _generate_detailed_metrics(self, metrics: Dict[str, Any]) -> str:
        """Сгенерировать детальные метрики для одного алгоритма."""
        output = []
        
        # Группируем метрики
        groups = {
            'Reward Metrics': [
                'mean_reward', 'std_reward', 'min_reward', 'max_reward', 'median_reward'
            ],
            'Success Metrics': [
                'success_rate', 'total_successes', 'total_episodes'
            ],
            'Collision Metrics': [
                'collision_rate', 'mean_collisions_per_episode', 'total_collisions'
            ],
            'Episode Metrics': [
                'mean_episode_length', 'std_episode_length',
                'min_episode_length', 'max_episode_length'
            ],
            'Path Metrics': [
                'mean_path_length', 'std_path_length',
                'mean_path_efficiency', 'std_path_efficiency'
            ],
            'Timing Metrics': [
                'mean_interception_time', 'std_interception_time', 'min_interception_time'
            ],
        }
        
        for group_name, metric_keys in groups.items():
            group_metrics = {k: metrics.get(k) for k in metric_keys if k in metrics}
            
            if group_metrics:
                output.append(f"**{group_name}:**\n")
                for key, value in group_metrics.items():
                    formatted_key = key.replace('_', ' ').title()
                    if isinstance(value, float):
                        if 'rate' in key or 'efficiency' in key:
                            output.append(f"- {formatted_key}: {value:.2%}\n")
                        else:
                            output.append(f"- {formatted_key}: {value:.3f}\n")
                    else:
                        output.append(f"- {formatted_key}: {value}\n")
                output.append("\n")
        
        return "".join(output)
    
    def _generate_plots_section(self, plots_dir: Path) -> str:
        """Сгенерировать секцию с визуализациями."""
        output = []
        
        # Ищем изображения
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif']
        images = []
        for ext in image_extensions:
            images.extend(plots_dir.glob(f'*{ext}'))
        
        if not images:
            return "*No visualizations available*\n"
        
        # Группируем по типам
        plot_groups = {
            'Training Metrics': ['reward', 'training', 'metric'],
            'Comparisons': ['comparison', 'box', 'violin', 'radar'],
            'Trajectories': ['trajectory', 'path'],
            'Heatmaps': ['heatmap', 'heat'],
            'Animations': ['animation', '.gif', '.mp4'],
        }
        
        for group_name, keywords in plot_groups.items():
            group_images = [
                img for img in images
                if any(kw in img.name.lower() for kw in keywords)
            ]
            
            if group_images:
                output.append(f"### {group_name}\n\n")
                for img in sorted(group_images):
                    rel_path = img.relative_to(plots_dir.parent)
                    output.append(f"![{img.stem}]({rel_path})\n\n")
        
        return "".join(output)
    
    def _generate_recommendations(
        self,
        algorithms: List[str],
        metrics_dict: Dict[str, Dict[str, Any]],
    ) -> str:
        """Сгенерировать рекомендации на основе результатов."""
        output = []
        
        # Находим лучший по разным критериям
        best_reward_algo = max(
            algorithms,
            key=lambda a: metrics_dict.get(a, {}).get('mean_reward', float('-inf'))
        )
        best_reward = metrics_dict[best_reward_algo].get('mean_reward', 0)
        
        best_success_algo = max(
            algorithms,
            key=lambda a: metrics_dict.get(a, {}).get('success_rate', 0)
        )
        best_success = metrics_dict[best_success_algo].get('success_rate', 0)
        
        output.append("### Best Performing Algorithms\n\n")
        output.append(f"- **Highest Reward:** {best_reward_algo.upper()} "
                     f"(mean reward: {best_reward:.2f})\n")
        output.append(f"- **Highest Success Rate:** {best_success_algo.upper()} "
                     f"(success rate: {best_success:.2%})\n")
        output.append("\n")
        
        # Анализ эффективности
        output.append("### Efficiency Analysis\n\n")
        
        for algo in algorithms:
            metrics = metrics_dict.get(algo, {})
            path_eff = metrics.get('mean_path_efficiency', 0)
            collision_rate = metrics.get('collision_rate', 0)
            
            if path_eff > 0.8:
                output.append(f"- **{algo.upper()}:** Excellent path efficiency ({path_eff:.2%})\n")
            elif path_eff > 0.6:
                output.append(f"- **{algo.upper()}:** Good path efficiency ({path_eff:.2%})\n")
            else:
                output.append(f"- **{algo.upper()}:** Needs improvement in path planning ({path_eff:.2%})\n")
            
            if collision_rate > 0.3:
                output.append(f"  - ⚠️ High collision rate ({collision_rate:.2%}) - "
                             f"consider tuning collision penalty\n")
        
        output.append("\n")
        
        # Общие рекомендации
        output.append("### General Recommendations\n\n")
        output.append("1. **For Best Performance:** Use the algorithm with highest success rate\n")
        output.append("2. **For Exploration:** Algorithms with higher path diversity may discover better strategies\n")
        output.append("3. **For Safety:** Choose algorithms with lower collision rates\n")
        output.append("4. **For Efficiency:** Prioritize algorithms with high path efficiency\n")
        
        return "".join(output)
    
    def generate_single_algorithm_report(
        self,
        algorithm: str,
        metrics: Dict[str, Any],
        config: Dict[str, Any] = None,
        plots_dir: Path = None,
    ) -> str:
        """
        Сгенерировать отчет для одного алгоритма.
        
        Args:
            algorithm: Название алгоритма
            metrics: Словарь с метриками
            config: Конфигурация
            plots_dir: Директория с графиками
        
        Returns:
            Путь к сгенерированному отчету
        """
        report_path = self.output_dir / f"{algorithm}_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {algorithm.upper()} Training Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            if config:
                f.write("## Configuration\n\n")
                f.write("```yaml\n")
                f.write(self._dict_to_yaml(config))
                f.write("```\n\n")
                f.write("---\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write(self._generate_detailed_metrics(metrics))
            f.write("\n---\n\n")
            
            if plots_dir and plots_dir.exists():
                f.write("## Visualizations\n\n")
                f.write(self._generate_plots_section(plots_dir))
                f.write("\n---\n\n")
            
            f.write("*Report generated automatically by RL-Interception framework*\n")
        
        return str(report_path)

