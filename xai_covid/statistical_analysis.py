import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, ttest_ind
from typing import Dict, List, Tuple
import pandas as pd
from config import Config
import json
import os

class StatisticalAnalyzer:
    def __init__(self, alpha=Config.STATISTICAL_PARAMS['alpha']):
        self.alpha = alpha
        self.bonferroni_correction = Config.STATISTICAL_PARAMS['bonferroni_correction']
        self.confidence_level = Config.STATISTICAL_PARAMS['confidence_level']
        self.results = {}
        self.rng = np.random.default_rng(Config.SEED)
    
    def wilcoxon_test(self, sample1: np.ndarray, sample2: np.ndarray, 
                      test_name: str = "comparison") -> Dict:
        statistic, p_value = wilcoxon(sample1, sample2, alternative='two-sided')
        
        adjusted_alpha = self.alpha
        if self.bonferroni_correction:
            adjusted_alpha = self.alpha / len(sample1)
        
        significant = p_value < adjusted_alpha
        
        effect_size = self.compute_cohens_d(sample1, sample2)
        
        return {
            'test_name': test_name,
            'test_type': 'Wilcoxon Signed-Rank',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': adjusted_alpha,
            'significant': significant,
            'effect_size': effect_size,
            'sample_size': len(sample1),
            'mean_diff': float(np.mean(sample1) - np.mean(sample2))
        }
    
    def mann_whitney_test(self, sample1: np.ndarray, sample2: np.ndarray,
                         test_name: str = "comparison") -> Dict:
        statistic, p_value = mannwhitneyu(sample1, sample2, alternative='two-sided')
        
        adjusted_alpha = self.alpha
        if self.bonferroni_correction:
            adjusted_alpha = self.alpha / 2
        
        significant = p_value < adjusted_alpha
        
        effect_size = self.compute_glass_delta(sample1, sample2)
        
        return {
            'test_name': test_name,
            'test_type': 'Mann-Whitney U',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': adjusted_alpha,
            'significant': significant,
            'effect_size': effect_size,
            'sample_size_1': len(sample1),
            'sample_size_2': len(sample2),
            'median_diff': float(np.median(sample1) - np.median(sample2))
        }
    
    def paired_t_test(self, sample1: np.ndarray, sample2: np.ndarray,
                     test_name: str = "comparison") -> Dict:
        statistic, p_value = ttest_rel(sample1, sample2)
        
        adjusted_alpha = self.alpha
        if self.bonferroni_correction:
            adjusted_alpha = self.alpha / len(sample1)
        
        significant = p_value < adjusted_alpha
        
        effect_size = self.compute_cohens_d(sample1, sample2)
        
        ci_lower, ci_upper = self.compute_confidence_interval(sample1 - sample2)
        
        return {
            'test_name': test_name,
            'test_type': 'Paired t-test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': adjusted_alpha,
            'significant': significant,
            'effect_size': effect_size,
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'sample_size': len(sample1)
        }
    
    def independent_t_test(self, sample1: np.ndarray, sample2: np.ndarray,
                          test_name: str = "comparison") -> Dict:
        statistic, p_value = ttest_ind(sample1, sample2)
        
        adjusted_alpha = self.alpha
        if self.bonferroni_correction:
            adjusted_alpha = self.alpha / 2
        
        significant = p_value < adjusted_alpha
        
        effect_size = self.compute_cohens_d(sample1, sample2)
        
        return {
            'test_name': test_name,
            'test_type': 'Independent t-test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': adjusted_alpha,
            'significant': significant,
            'effect_size': effect_size,
            'sample_size_1': len(sample1),
            'sample_size_2': len(sample2)
        }
    
    def compute_cohens_d(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        n1, n2 = len(sample1), len(sample2)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return float(cohens_d)
    
    def compute_glass_delta(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std2 = np.std(sample2, ddof=1)
        
        glass_delta = (mean1 - mean2) / std2 if std2 > 0 else 0
        
        return float(glass_delta)
    
    def compute_hedges_g(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        cohens_d = self.compute_cohens_d(sample1, sample2)
        n1, n2 = len(sample1), len(sample2)
        
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction_factor
        
        return float(hedges_g)
    
    def compute_confidence_interval(self, data: np.ndarray, confidence=None) -> Tuple[float, float]:
        if confidence is None:
            confidence = self.confidence_level
        
        mean = np.mean(data)
        se = stats.sem(data)
        interval = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return mean - interval, mean + interval
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                     statistic_func=np.mean,
                                     n_iterations=None) -> Tuple[float, float]:
        if n_iterations is None:
            n_iterations = Config.STATISTICAL_PARAMS['bootstrap_iterations']
        
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_iterations):
            sample = self.rng.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return float(lower), float(upper)
    
    def compare_multiple_methods(self, results_dict: Dict[str, np.ndarray],
                                baseline_method: str = None) -> Dict:
        methods = list(results_dict.keys())
        
        if baseline_method is None:
            baseline_method = methods[0]
        
        comparisons = {}
        num_comparisons = len(methods) - 1
        
        adjusted_alpha = self.alpha
        if self.bonferroni_correction:
            adjusted_alpha = self.alpha / num_comparisons
        
        baseline_data = results_dict[baseline_method]
        
        for method in methods:
            if method == baseline_method:
                continue
            
            method_data = results_dict[method]
            
            test_result = self.wilcoxon_test(
                method_data, baseline_data,
                test_name=f"{method}_vs_{baseline_method}"
            )
            test_result['adjusted_alpha'] = adjusted_alpha
            test_result['significant_after_correction'] = test_result['p_value'] < adjusted_alpha
            
            comparisons[f"{method}_vs_{baseline_method}"] = test_result
        
        summary = self._generate_comparison_summary(comparisons, baseline_method)
        
        return {
            'baseline_method': baseline_method,
            'num_comparisons': num_comparisons,
            'adjusted_alpha': adjusted_alpha,
            'comparisons': comparisons,
            'summary': summary
        }
    
    def _generate_comparison_summary(self, comparisons: Dict, baseline: str) -> Dict:
        significant_improvements = []
        significant_degradations = []
        
        for comp_name, comp_result in comparisons.items():
            if comp_result['significant_after_correction']:
                if comp_result['mean_diff'] > 0:
                    significant_improvements.append(comp_name)
                else:
                    significant_degradations.append(comp_name)
        
        return {
            'num_significant_improvements': len(significant_improvements),
            'num_significant_degradations': len(significant_degradations),
            'significant_improvements': significant_improvements,
            'significant_degradations': significant_degradations
        }
    
    def friedman_test(self, *samples, test_name: str = "friedman") -> Dict:
        statistic, p_value = stats.friedmanchisquare(*samples)
        
        significant = p_value < self.alpha
        
        return {
            'test_name': test_name,
            'test_type': 'Friedman Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'significant': significant,
            'num_groups': len(samples),
            'sample_size': len(samples[0])
        }
    
    def anova_test(self, *samples, test_name: str = "anova") -> Dict:
        statistic, p_value = stats.f_oneway(*samples)
        
        significant = p_value < self.alpha
        
        return {
            'test_name': test_name,
            'test_type': 'One-way ANOVA',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'significant': significant,
            'num_groups': len(samples)
        }
    
    def comprehensive_comparison(self, method_results: Dict[str, List[float]],
                                metric_name: str = "performance") -> Dict:
        comparison_results = {
            'metric_name': metric_name,
            'descriptive_statistics': {},
            'pairwise_comparisons': {},
            'overall_test': None,
            'rankings': {},
            'best_method': None
        }
        
        for method, results in method_results.items():
            data = np.array(results)
            mean_val = np.mean(data)
            
            ci_lower, ci_upper = self.compute_confidence_interval(data)
            boot_lower, boot_upper = self.bootstrap_confidence_interval(data)
            
            comparison_results['descriptive_statistics'][method] = {
                'mean': float(mean_val),
                'std': float(np.std(data)),
                'median': float(np.median(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'ci_95': (float(ci_lower), float(ci_upper)),
                'bootstrap_ci_95': (float(boot_lower), float(boot_upper)),
                'n_samples': len(data)
            }
        
        if len(method_results) > 2:
            samples = [np.array(results) for results in method_results.values()]
            comparison_results['overall_test'] = self.friedman_test(*samples, test_name=metric_name)
        
        methods = list(method_results.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = np.array(method_results[method1])
                data2 = np.array(method_results[method2])
                
                test_result = self.wilcoxon_test(data1, data2, test_name=f"{method1}_vs_{method2}")
                comparison_results['pairwise_comparisons'][f"{method1}_vs_{method2}"] = test_result
        
        mean_scores = {method: stats['mean'] 
                      for method, stats in comparison_results['descriptive_statistics'].items()}
        ranked_methods = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (method, score) in enumerate(ranked_methods, 1):
            comparison_results['rankings'][method] = {
                'rank': rank,
                'score': score
            }
        
        comparison_results['best_method'] = ranked_methods[0][0]
        
        return comparison_results
    
    def save_results(self, filename: str):
        filepath = os.path.join(Config.STATISTICAL_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Statistical results saved to: {filepath}")
    
    def generate_latex_table(self, comparison_results: Dict) -> str:
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\caption{Statistical Comparison of Methods}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\hline\n"
        latex += "Method & Mean & Std & 95\\% CI & p-value \\\\\n"
        latex += "\\hline\n"
        
        for method, stats in comparison_results['descriptive_statistics'].items():
            ci_lower, ci_upper = stats['ci_95']
            latex += f"{method} & {stats['mean']:.4f} & {stats['std']:.4f} & "
            latex += f"[{ci_lower:.4f}, {ci_upper:.4f}] & - \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def power_analysis(self, sample1: np.ndarray, sample2: np.ndarray,
                      alpha: float = None) -> Dict:
        if alpha is None:
            alpha = self.alpha
        
        effect_size = self.compute_cohens_d(sample1, sample2)
        n = len(sample1)
        
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(n/2) - z_alpha
        power = norm.cdf(z_beta)
        
        return {
            'effect_size': float(effect_size),
            'sample_size': n,
            'alpha': alpha,
            'power': float(power),
            'adequate_power': power >= 0.8
        }
    
    def generate_report(self, all_results: Dict) -> str:
        report = "# Statistical Analysis Report\n\n"
        report += f"## Configuration\n"
        report += f"- Significance Level (α): {self.alpha}\n"
        report += f"- Confidence Level: {self.confidence_level}\n"
        report += f"- Bonferroni Correction: {self.bonferroni_correction}\n\n"
        
        report += "## Summary of Findings\n\n"
        
        for analysis_name, results in all_results.items():
            report += f"### {analysis_name}\n\n"
            
            if 'best_method' in results:
                report += f"**Best Method:** {results['best_method']}\n\n"
            
            if 'descriptive_statistics' in results:
                report += "#### Descriptive Statistics\n\n"
                for method, stats in results['descriptive_statistics'].items():
                    report += f"- **{method}**: Mean={stats['mean']:.4f} ± {stats['std']:.4f}\n"
                report += "\n"
            
            if 'pairwise_comparisons' in results:
                report += "#### Pairwise Comparisons\n\n"
                for comp_name, comp_result in results['pairwise_comparisons'].items():
                    sig_marker = "***" if comp_result['significant'] else ""
                    report += f"- {comp_name}: p={comp_result['p_value']:.6f} {sig_marker}\n"
                report += "\n"
        
        return report
