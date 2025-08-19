# evaluation.py - 評估系統模塊
# search evaluation metrics and a/b testing framework

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
from datetime import datetime, timedelta
import json
from config import Config


class SearchEvaluator:
    """
    comprehensive search evaluation system
    demonstrates: metrics definition, measurement, analysis
    """

    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.evaluation_queries = []
        self.ground_truth = {}
        self.experiment_results = {}

        print("📊 search evaluator initialized")

    def create_evaluation_dataset(self):
        """create test queries with ground truth relevance"""
        print("🔬 creating evaluation dataset...")

        self.evaluation_queries = [
            {
                'query': 'artificial intelligence machine learning',
                'relevant_categories': ['technology'],
                'expected_keywords': ['ai', 'machine learning', 'llm', 'algorithm'],
                'query_type': 'technology'
            },
            {
                'query': 'climate change environment',
                'relevant_categories': ['science', 'health'],
                'expected_keywords': ['climate', 'environment', 'warming', 'carbon'],
                'query_type': 'environmental'
            },
            {
                'query': 'stock market investment finance',
                'relevant_categories': ['business'],
                'expected_keywords': ['stock', 'investment', 'market', 'financial'],
                'query_type': 'financial'
            },
            {
                'query': 'covid health medicine',
                'relevant_categories': ['health', 'science'],
                'expected_keywords': ['covid', 'health', 'medical', 'virus'],
                'query_type': 'medical'
            },
            {
                'query': 'football soccer sports',
                'relevant_categories': ['sports'],
                'expected_keywords': ['football', 'soccer', 'sports', 'team'],
                'query_type': 'sports'
            }
        ]

        print(f"✅ created {len(self.evaluation_queries)} evaluation queries")

        # generate ground truth relevance scores
        for query_data in self.evaluation_queries:
            query = query_data['query']
            self.ground_truth[query] = self._generate_relevance_scores(query_data)

        print("✅ ground truth relevance scores generated")

    def _generate_relevance_scores(self, query_data):
        """generate relevance scores for articles based on category and keyword matching"""
        relevance_scores = {}

        for i, article in enumerate(self.search_engine.articles):
            score = 0

            # category relevance
            if article['category'] in query_data['relevant_categories']:
                score += 2

            # keyword matching
            text = f"{article['title']} {article.get('content', '')}".lower()
            keyword_matches = sum(1 for keyword in query_data['expected_keywords']
                                  if keyword.lower() in text)
            score += keyword_matches

            # normalize to 0-3 scale
            relevance_scores[i] = min(score, 3)

        return relevance_scores

    def compute_ndcg(self, search_results, query, k=5):
        """compute normalized discounted cumulative gain"""
        if query not in self.ground_truth:
            return 0.0

        ground_truth = self.ground_truth[query]

        # get relevance scores for returned results
        relevances = []
        for result in search_results[:k]:
            article_id = result['article_id']
            relevance = ground_truth.get(article_id, 0)
            relevances.append(relevance)

        # compute dcg
        dcg = 0.0
        for i, rel in enumerate(relevances):
            dcg += (2 ** rel - 1) / np.log2(i + 2)

        # compute ideal dcg
        ideal_relevances = sorted(ground_truth.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances):
            idcg += (2 ** rel - 1) / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def compute_precision_at_k(self, search_results, query, k=5):
        """compute precision@k"""
        if query not in self.ground_truth:
            return 0.0

        ground_truth = self.ground_truth[query]

        relevant_in_topk = 0
        for result in search_results[:k]:
            article_id = result['article_id']
            if ground_truth.get(article_id, 0) >= 2:
                relevant_in_topk += 1

        return relevant_in_topk / k

    def evaluate_search_method(self, method, k=5):
        """comprehensive evaluation of a search method"""
        print(f"📊 evaluating {method} search method...")

        results = {
            'method': method,
            'ndcg_scores': [],
            'precision_scores': [],
            'response_times': [],
            'query_details': []
        }

        for query_data in self.evaluation_queries:
            query = query_data['query']

            # measure response time
            start_time = datetime.now()
            search_results = self.search_engine.search(query, method=method, k=k * 2)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            # compute metrics
            ndcg = self.compute_ndcg(search_results, query, k)
            precision = self.compute_precision_at_k(search_results, query, k)

            results['ndcg_scores'].append(ndcg)
            results['precision_scores'].append(precision)
            results['response_times'].append(response_time)
            results['query_details'].append({
                'query': query,
                'query_type': query_data['query_type'],
                'ndcg': ndcg,
                'precision': precision,
                'response_time': response_time,
                'results_count': len(search_results)
            })

        # compute summary statistics
        results['avg_ndcg'] = np.mean(results['ndcg_scores'])
        results['avg_precision'] = np.mean(results['precision_scores'])
        results['avg_response_time'] = np.mean(results['response_times'])

        print(f"   ✅ {method} evaluation complete:")
        print(f"      avg ndcg@{k}: {results['avg_ndcg']:.4f}")
        print(f"      avg precision@{k}: {results['avg_precision']:.4f}")
        print(f"      avg response time: {results['avg_response_time']:.3f}s")

        return results


class ABTestFramework:
    """
    a/b testing framework for search algorithms
    demonstrates: experimental design, statistical analysis, causal inference
    """

    def __init__(self, search_evaluator):
        self.evaluator = search_evaluator
        self.experiments = {}
        print("🧪 a/b testing framework initialized")

    def run_ab_test(self, control_method, treatment_method, test_name, k=5):
        """run a/b test comparing two search methods"""
        print(f"\n🧪 running a/b test: {test_name}")
        print(f"   control: {control_method} vs treatment: {treatment_method}")

        # evaluate both methods
        control_results = self.evaluator.evaluate_search_method(control_method, k)
        treatment_results = self.evaluator.evaluate_search_method(treatment_method, k)

        # statistical analysis
        analysis = self._analyze_ab_results(control_results, treatment_results, test_name)

        # store experiment
        self.experiments[test_name] = {
            'control': control_results,
            'treatment': treatment_results,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }

        return analysis

    def _analyze_ab_results(self, control, treatment, test_name):
        """statistical analysis of a/b test results"""
        print(f"📊 analyzing a/b test results for {test_name}...")

        # paired t-tests for metrics
        ndcg_ttest = stats.ttest_rel(treatment['ndcg_scores'], control['ndcg_scores'])
        precision_ttest = stats.ttest_rel(treatment['precision_scores'], control['precision_scores'])

        # effect sizes
        ndcg_improvement = (treatment['avg_ndcg'] - control['avg_ndcg']) / control['avg_ndcg'] * 100
        precision_improvement = (treatment['avg_precision'] - control['avg_precision']) / control['avg_precision'] * 100

        analysis = {
            'test_name': test_name,
            'sample_size': len(control['ndcg_scores']),
            'control_method': control['method'],
            'treatment_method': treatment['method'],

            # ndcg analysis
            'control_avg_ndcg': control['avg_ndcg'],
            'treatment_avg_ndcg': treatment['avg_ndcg'],
            'ndcg_improvement_pct': ndcg_improvement,
            'ndcg_p_value': ndcg_ttest.pvalue,
            'ndcg_statistically_significant': ndcg_ttest.pvalue < 0.05,

            # precision analysis
            'control_avg_precision': control['avg_precision'],
            'treatment_avg_precision': treatment['avg_precision'],
            'precision_improvement_pct': precision_improvement,
            'precision_p_value': precision_ttest.pvalue,
            'precision_statistically_significant': precision_ttest.pvalue < 0.05,

            # performance analysis
            'control_avg_response_time': control['avg_response_time'],
            'treatment_avg_response_time': treatment['avg_response_time'],
            'response_time_change_pct': (treatment['avg_response_time'] - control['avg_response_time']) / control[
                'avg_response_time'] * 100
        }

        # print results
        print(f"\n📋 a/b test results: {test_name}")
        print("=" * 50)
        print(f"📊 ndcg results:")
        print(f"   control ({control['method']}): {analysis['control_avg_ndcg']:.4f}")
        print(f"   treatment ({treatment['method']}): {analysis['treatment_avg_ndcg']:.4f}")
        print(f"   improvement: {analysis['ndcg_improvement_pct']:+.1f}%")
        print(f"   p-value: {analysis['ndcg_p_value']:.4f}")
        print(f"   statistically significant: {'✅ yes' if analysis['ndcg_statistically_significant'] else '❌ no'}")

        print(f"\n📊 precision results:")
        print(f"   control: {analysis['control_avg_precision']:.4f}")
        print(f"   treatment: {analysis['treatment_avg_precision']:.4f}")
        print(f"   improvement: {analysis['precision_improvement_pct']:+.1f}%")
        print(f"   p-value: {analysis['precision_p_value']:.4f}")
        print(f"   statistically significant: {'✅ yes' if analysis['precision_statistically_significant'] else '❌ no'}")

        print(f"\n⚡ performance impact:")
        print(f"   response time change: {analysis['response_time_change_pct']:+.1f}%")

        # recommendation
        if analysis['ndcg_statistically_significant'] and analysis['ndcg_improvement_pct'] > 0:
            recommendation = f"✅ recommend adopting {treatment['method']} - significant quality improvement"
        elif analysis['precision_statistically_significant'] and analysis['precision_improvement_pct'] > 0:
            recommendation = f"✅ recommend adopting {treatment['method']} - significant precision improvement"
        else:
            recommendation = f"⚠️  insufficient evidence to recommend {treatment['method']}"

        print(f"\n💡 recommendation: {recommendation}")

        return analysis

    def generate_experiment_report(self):
        """generate comprehensive experiment report"""
        print(f"\n📋 comprehensive a/b testing report")
        print("=" * 60)

        if not self.experiments:
            print("❌ no experiments conducted yet")
            return

        for exp_name, exp_data in self.experiments.items():
            analysis = exp_data['analysis']

            print(f"\n🧪 experiment: {exp_name}")
            print(f"   control: {analysis['control_method']}")
            print(f"   treatment: {analysis['treatment_method']}")
            print(
                f"   ndcg improvement: {analysis['ndcg_improvement_pct']:+.1f}% (sig: {analysis['ndcg_statistically_significant']})")
            print(
                f"   precision improvement: {analysis['precision_improvement_pct']:+.1f}% (sig: {analysis['precision_statistically_significant']})")

        print(f"\n🎯 key insights:")
        print("   - hybrid search generally outperforms single methods")
        print("   - semantic search helps with vocabulary mismatch")
        print("   - statistical significance requires sufficient test queries")

        return self.experiments


class UserBehaviorAnalyzer:
    """
    user behavior analysis system for click-through rate modeling
    demonstrates: user research, behavior modeling
    """

    def __init__(self):
        self.click_logs = []
        self.search_sessions = []
        print("👥 user behavior analyzer initialized")

    def simulate_user_clicks(self, search_results, query_intent):
        """simulate realistic user click behavior"""
        click_probabilities = []

        for i, result in enumerate(search_results):
            # position bias
            position_bias = 1.0 / (i + 1) ** 0.5

            # relevance bias
            relevance_bias = result.get('score', result.get('hybrid_score', 0.5))

            # query-result matching
            title_match = self.compute_title_query_match(result['title'], query_intent)

            click_prob = position_bias * relevance_bias * title_match
            click_probabilities.append(min(click_prob, 0.8))

        return click_probabilities

    def compute_title_query_match(self, title, query):
        """compute semantic match between title and query"""
        title_words = set(title.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.5

        intersection = title_words & query_words
        return len(intersection) / len(query_words)

    def analyze_search_patterns(self, search_logs):
        """analyze search patterns for insights"""
        if not search_logs:
            return {}

        patterns = {
            'avg_query_length': np.mean([len(log['query'].split()) for log in search_logs]),
            'popular_categories': self.get_category_distribution(search_logs),
            'peak_search_times': self.get_temporal_patterns(search_logs),
            'user_satisfaction_score': self.compute_satisfaction_score(search_logs)
        }

        return patterns

    def get_category_distribution(self, search_logs):
        """get distribution of searched categories"""
        category_counts = {}

        for log in search_logs:
            for category in log.get('top_categories', []):
                category_counts[category] = category_counts.get(category, 0) + 1

        return dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))

    def get_temporal_patterns(self, search_logs):
        """analyze temporal search patterns"""
        # simplified temporal analysis
        hours = [12, 14, 16, 18, 20]  # simulated peak hours
        return {f"{hour}:00": np.random.randint(50, 200) for hour in hours}

    def compute_satisfaction_score(self, search_logs):
        """compute overall user satisfaction score"""
        if not search_logs:
            return 0.5

        # simplified satisfaction based on results count
        avg_results = np.mean([log['results_count'] for log in search_logs if 'results_count' in log])
        return min(avg_results / 10, 1.0)  # normalize to 0-1


# standalone execution for testing
if __name__ == "__main__":
    config = Config()

    print("🧪 testing evaluation system...")
    print("   (requires search engine to be initialized)")
    print("   run main.py for complete pipeline test")

    # test user behavior analyzer
    behavior_analyzer = UserBehaviorAnalyzer()

    # simulate some search logs
    test_logs = [
        {'query': 'artificial intelligence', 'results_count': 8, 'top_categories': ['technology']},
        {'query': 'climate change', 'results_count': 6, 'top_categories': ['science']},
        {'query': 'stock market', 'results_count': 10, 'top_categories': ['business']}
    ]

    patterns = behavior_analyzer.analyze_search_patterns(test_logs)
    print("📊 search patterns analysis:")
    for key, value in patterns.items():
        print(f"   {key}: {value}")

    print("✅ evaluation system module loaded successfully")