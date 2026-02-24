import json
import numpy as np
import pandas as pd
from typing import Dict, List
import os
from config import Config
from statistical_analysis import StatisticalAnalyzer

class ResultsAnalyzer:
    def __init__(self):
        self.all_results = {}
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def load_all_results(self):
        results_dir = Config.RESULTS_DIR
        
        result_files = {
            'cross_dataset': 'cross_dataset_validation_results.json',
            'ablation': os.path.join(Config.ABLATION_DIR, 'complete_ablation_results.json'),
            'sota_comparison': 'sota_baseline_comparison.json',
            'xai_metrics': 'comprehensive_xai_metrics.json'
        }
        
        for key, filename in result_files.items():
            filepath = os.path.join(results_dir, filename) if not os.path.isabs(filename) else filename
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.all_results[key] = json.load(f)
                print(f"Loaded: {key}")
            else:
                print(f"Warning: {filepath} not found")
    
    def compile_performance_metrics(self) -> Dict:
        metrics = {
            'model_performance': {},
            'xai_quality': {},
            'cross_dataset_generalization': {},
            'computational_efficiency': {}
        }
        
        if 'sota_comparison' in self.all_results:
            for method, results in self.all_results['sota_comparison'].items():
                if 'model_performance' in results:
                    metrics['model_performance'][method] = results['model_performance']
                if 'xai_quality' in results:
                    metrics['xai_quality'][method] = results['xai_quality']
        
        if 'cross_dataset' in self.all_results:
            summary = self.all_results['cross_dataset'].get('summary', {})
            metrics['cross_dataset_generalization'] = {
                'average_accuracy': summary.get('average_accuracy', 0),
                'average_f1': summary.get('average_f1', 0),
                'average_consistency': summary.get('average_xai_consistency', 0)
            }
        
        return metrics
    
    def generate_latex_tables(self) -> Dict[str, str]:
        tables = {}
        
        if 'model_performance' in self.compile_performance_metrics():
            tables['performance'] = self._generate_performance_table()
        
        if 'ablation' in self.all_results:
            tables['ablation'] = self._generate_ablation_table()
        
        if 'xai_quality' in self.compile_performance_metrics():
            tables['xai_comparison'] = self._generate_xai_table()
        
        for name, latex in tables.items():
            filepath = os.path.join(Config.RESULTS_DIR, f'table_{name}.tex')
            with open(filepath, 'w') as f:
                f.write(latex)
            print(f"LaTeX table saved: {filepath}")
        
        return tables
    
    def _generate_performance_table(self) -> str:
        metrics = self.compile_performance_metrics()['model_performance']
        
        latex = "\\begin{table}[ht]\n\\centering\n"
        latex += "\\caption{Model Performance Comparison}\n"
        latex += "\\label{tab:performance}\n"
        latex += "\\begin{tabular}{lccccc}\n"
        latex += "\\toprule\n"
        latex += "Method & Accuracy & Precision & Recall & F1-Score & AUC \\\\\n"
        latex += "\\midrule\n"
        
        for method, perf in metrics.items():
            method_name = method.replace('_', ' ').title()
            latex += f"{method_name} & "
            latex += f"{perf.get('accuracy', 0):.4f} & "
            latex += f"{perf.get('precision', 0):.4f} & "
            latex += f"{perf.get('recall', 0):.4f} & "
            latex += f"{perf.get('f1', 0):.4f} & "
            latex += f"{perf.get('auc', 0):.4f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _generate_ablation_table(self) -> str:
        ablation_results = self.all_results['ablation']
        
        latex = "\\begin{table}[ht]\n\\centering\n"
        latex += "\\caption{Ablation Study Results}\n"
        latex += "\\label{tab:ablation}\n"
        latex += "\\begin{tabular}{lcc}\n"
        latex += "\\toprule\n"
        latex += "Configuration & Accuracy & Performance Drop (\\%) \\\\\n"
        latex += "\\midrule\n"
        
        if 'component_ablation' in ablation_results:
            baseline = ablation_results['component_ablation']['baseline_performance']
            latex += f"Baseline (All Components) & {baseline['mean_accuracy']:.4f} & - \\\\\n"
            
            for config, results in ablation_results['component_ablation']['ablated_results'].items():
                if config != 'baseline_all_components':
                    drop_pct = ablation_results['component_ablation']['performance_drops'][config]['relative_drop_percent']
                    latex += f"{config.replace('_', ' ').title()} & "
                    latex += f"{results['mean_accuracy']:.4f} & "
                    latex += f"{drop_pct:.2f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _generate_xai_table(self) -> str:
        metrics = self.compile_performance_metrics()['xai_quality']
        
        latex = "\\begin{table}[ht]\n\\centering\n"
        latex += "\\caption{XAI Quality Metrics}\n"
        latex += "\\label{tab:xai}\n"
        latex += "\\begin{tabular}{lcc}\n"
        latex += "\\toprule\n"
        latex += "Method & Faithfulness & Stability \\\\\n"
        latex += "\\midrule\n"
        
        for method, xai in metrics.items():
            method_name = method.replace('_', ' ').title()
            latex += f"{method_name} & "
            latex += f"{xai.get('faithfulness', 0):.4f} $\\pm$ {xai.get('faithfulness_std', 0):.4f} & "
            latex += f"{xai.get('stability', 0):.4f} $\\pm$ {xai.get('stability_std', 0):.4f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def generate_summary_statistics(self) -> Dict:
        summary = {
            'total_experiments': 0,
            'key_findings': [],
            'performance_highlights': {},
            'statistical_significance': {}
        }
        
        metrics = self.compile_performance_metrics()
        
        if 'model_performance' in metrics:
            accuracies = [perf['accuracy'] for perf in metrics['model_performance'].values()]
            summary['performance_highlights']['best_accuracy'] = max(accuracies)
            summary['performance_highlights']['mean_accuracy'] = np.mean(accuracies)
            summary['performance_highlights']['std_accuracy'] = np.std(accuracies)
        
        if 'xai_quality' in metrics:
            faithfulness_scores = [xai['faithfulness'] for xai in metrics['xai_quality'].values()]
            summary['performance_highlights']['best_faithfulness'] = max(faithfulness_scores)
            summary['performance_highlights']['mean_faithfulness'] = np.mean(faithfulness_scores)
        
        if 'ablation' in self.all_results:
            summary['total_experiments'] += len(self.all_results['ablation'].get('component_ablation', {}).get('ablated_results', {}))
        
        if 'cross_dataset_generalization' in metrics:
            summary['performance_highlights']['cross_dataset_accuracy'] = metrics['cross_dataset_generalization']['average_accuracy']
        
        summary['key_findings'] = self._extract_key_findings(metrics)
        
        return summary
    
    def _extract_key_findings(self, metrics: Dict) -> List[str]:
        findings = []
        
        if 'model_performance' in metrics:
            best_method = max(metrics['model_performance'].items(), 
                            key=lambda x: x[1]['accuracy'])
            findings.append(f"Best performing method: {best_method[0]} with {best_method[1]['accuracy']:.4f} accuracy")
        
        if 'xai_quality' in metrics:
            best_xai = max(metrics['xai_quality'].items(),
                          key=lambda x: x[1]['faithfulness'])
            findings.append(f"Best XAI faithfulness: {best_xai[0]} with score {best_xai[1]['faithfulness']:.4f}")
        
        if 'cross_dataset_generalization' in metrics:
            gen_acc = metrics['cross_dataset_generalization']['average_accuracy']
            if gen_acc > 0.85:
                findings.append(f"Strong cross-dataset generalization: {gen_acc:.4f} average accuracy")
        
        return findings
    
    def calculate_publication_metrics(self) -> Dict:
        pub_metrics = {
            'meets_cvpr_standards': False,
            'checklist': {},
            'recommendations': []
        }
        
        metrics = self.compile_performance_metrics()
        
        xai_faithfulness = 0
        xai_stability = 0
        if 'xai_quality' in metrics and 'proposed_method' in metrics['xai_quality']:
            xai_faithfulness = metrics['xai_quality']['proposed_method'].get('faithfulness', 0)
            xai_stability = metrics['xai_quality']['proposed_method'].get('stability', 0)
        
        pub_metrics['checklist']['xai_faithfulness'] = xai_faithfulness >= Config.XAI_EVALUATION_PARAMS['min_faithfulness_target']
        pub_metrics['checklist']['xai_stability'] = xai_stability >= Config.XAI_EVALUATION_PARAMS['min_stability_target']
        
        cross_dataset_drop = 0
        if 'cross_dataset_generalization' in metrics:
            cross_dataset_drop = 1.0 - metrics['cross_dataset_generalization']['average_accuracy']
        
        pub_metrics['checklist']['cross_dataset_performance'] = cross_dataset_drop <= Config.CROSS_DATASET_PARAMS['max_performance_drop']
        
        pub_metrics['checklist']['statistical_significance'] = 'statistical_analysis' in self.all_results
        
        pub_metrics['checklist']['ablation_study'] = 'ablation' in self.all_results
        
        pub_metrics['checklist']['sota_comparison'] = 'sota_comparison' in self.all_results
        
        pub_metrics['meets_cvpr_standards'] = all(pub_metrics['checklist'].values())
        
        if not pub_metrics['checklist']['xai_faithfulness']:
            pub_metrics['recommendations'].append(f"Improve XAI faithfulness (current: {xai_faithfulness:.4f}, target: {Config.XAI_EVALUATION_PARAMS['min_faithfulness_target']:.4f})")
        
        if not pub_metrics['checklist']['xai_stability']:
            pub_metrics['recommendations'].append(f"Improve XAI stability (current: {xai_stability:.4f}, target: {Config.XAI_EVALUATION_PARAMS['min_stability_target']:.4f})")
        
        if not pub_metrics['checklist']['cross_dataset_performance']:
            pub_metrics['recommendations'].append(f"Reduce cross-dataset performance drop (current: {cross_dataset_drop*100:.2f}%, target: <{Config.CROSS_DATASET_PARAMS['max_performance_drop']*100:.2f}%)")
        
        return pub_metrics
    
    def generate_comprehensive_report(self) -> str:
        self.load_all_results()
        
        summary = self.generate_summary_statistics()
        pub_metrics = self.calculate_publication_metrics()
        
        report = "# Comprehensive Results Analysis Report\n\n"
        report += "## Executive Summary\n\n"
        report += f"Total experiments conducted: {summary['total_experiments']}\n\n"
        
        report += "### Key Findings\n\n"
        for finding in summary['key_findings']:
            report += f"- {finding}\n"
        report += "\n"
        
        report += "## Performance Highlights\n\n"
        for metric, value in summary['performance_highlights'].items():
            report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
        report += "\n"
        
        report += "## Publication Readiness Assessment\n\n"
        report += f"**Meets CVPR 2026 Standards**: {'✓ Yes' if pub_metrics['meets_cvpr_standards'] else '✗ No'}\n\n"
        
        report += "### Checklist:\n\n"
        for criterion, passed in pub_metrics['checklist'].items():
            status = "✓" if passed else "✗"
            report += f"- [{status}] {criterion.replace('_', ' ').title()}\n"
        report += "\n"
        
        if pub_metrics['recommendations']:
            report += "### Recommendations for Improvement:\n\n"
            for rec in pub_metrics['recommendations']:
                report += f"- {rec}\n"
            report += "\n"
        
        report += "## Detailed Results\n\n"
        
        metrics = self.compile_performance_metrics()
        
        if 'model_performance' in metrics:
            report += "### Model Performance\n\n"
            report += "| Method | Accuracy | F1-Score | AUC |\n"
            report += "|--------|----------|----------|-----|\n"
            for method, perf in metrics['model_performance'].items():
                report += f"| {method} | {perf.get('accuracy', 0):.4f} | {perf.get('f1', 0):.4f} | {perf.get('auc', 0):.4f} |\n"
            report += "\n"
        
        if 'xai_quality' in metrics:
            report += "### XAI Quality Metrics\n\n"
            report += "| Method | Faithfulness | Stability |\n"
            report += "|--------|--------------|----------|\n"
            for method, xai in metrics['xai_quality'].items():
                report += f"| {method} | {xai.get('faithfulness', 0):.4f} | {xai.get('stability', 0):.4f} |\n"
            report += "\n"
        
        report_path = os.path.join(Config.RESULTS_DIR, 'comprehensive_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nComprehensive report saved to: {report_path}")
        
        return report
    
    def export_results_to_csv(self):
        metrics = self.compile_performance_metrics()
        
        if 'model_performance' in metrics:
            df = pd.DataFrame.from_dict(metrics['model_performance'], orient='index')
            csv_path = os.path.join(Config.RESULTS_DIR, 'model_performance.csv')
            df.to_csv(csv_path)
            print(f"Model performance exported to: {csv_path}")
        
        if 'xai_quality' in metrics:
            df = pd.DataFrame.from_dict(metrics['xai_quality'], orient='index')
            csv_path = os.path.join(Config.RESULTS_DIR, 'xai_quality.csv')
            df.to_csv(csv_path)
            print(f"XAI quality exported to: {csv_path}")

if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    report = analyzer.generate_comprehensive_report()
    analyzer.generate_latex_tables()
    analyzer.export_results_to_csv()
    
    print("\n" + "="*60)
    print("Results analysis complete!")
    print("="*60)