#!/usr/bin/env python3
# main.py - 主程序文件
# complete multimodal search engine pipeline orchestrator

import argparse
import os
import sys
import json
import time
from datetime import datetime
import logging

# import our modules
from config import Config
from data_collector import NewsDataCollector, ScalableDataCollector
from search_engine import MultiModalSearchEngine, ScalableSearchEngine
from ranking_system import LearningToRankSystem
from evaluation import SearchEvaluator, ABTestFramework, UserBehaviorAnalyzer
from user_modeling import UserBehaviorSimulator, PersonalizedRanker, PersonalizedSearchEngine


def setup_logging():
    """setup logging configuration"""
    # create logs directory
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/search_engine.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def collect_data(config, logger):
    """data collection stage"""
    logger.info("🚀 Starting data collection...")

    # choose collector based on target size
    if config.TARGET_ARTICLES > 1000:
        collector = ScalableDataCollector(max_workers=config.MAX_WORKERS)
        articles = collector.collect_reddit_data(config.TARGET_ARTICLES)
    else:
        collector = NewsDataCollector(config.NEWS_API_KEY)
        articles = collector.collect_all_categories()

    # save collected data
    output_file = f"{config.DATA_DIR}/raw/articles_{datetime.now().strftime('%Y%m%d')}.json"

    if hasattr(collector, 'save_articles'):
        collector.save_articles(articles, output_file)
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Data collection complete: {len(articles)} articles saved to {output_file}")
    return output_file, articles


def build_search_indices(config, logger, articles_file):
    """build search indices stage"""
    logger.info("🔍 Building search indices...")

    # choose engine based on data size
    if config.TARGET_ARTICLES > 1000:
        search_engine = ScalableSearchEngine(
            chunk_size=config.CHUNK_SIZE,
            max_workers=config.MAX_WORKERS
        )
        search_engine.load_articles(articles_file)
        search_engine.build_scalable_indices()
    else:
        search_engine = MultiModalSearchEngine(config.OPENAI_API_KEY)
        search_engine.load_articles(articles_file)

    logger.info("✅ Search indices built successfully")
    return search_engine


def train_ranking_models(config, logger, search_engine):
    """train learning to rank models stage"""
    logger.info("🤖 Training ranking models...")

    # create evaluator for training data generation
    evaluator = SearchEvaluator(search_engine)
    evaluator.create_evaluation_dataset()

    # initialize and train l2r system
    l2r_system = LearningToRankSystem(search_engine, evaluator)
    training_data = l2r_system.generate_training_data()

    if training_data is not None and len(training_data) > 0:
        model_performance = l2r_system.train_models()
        logger.info("✅ Ranking models trained successfully")
        return l2r_system, evaluator, model_performance
    else:
        logger.warning("⚠️  No training data generated, skipping model training")
        return None, evaluator, None


def simulate_user_behavior(config, logger, articles):
    """simulate user behavior and interactions"""
    logger.info("👥 Simulating user behavior...")

    # generate user profiles
    user_simulator = UserBehaviorSimulator(
        num_users=config.NUM_USERS,
        num_articles=len(articles)
    )

    users_df = user_simulator.generate_user_profiles()
    interactions_df = user_simulator.simulate_search_interactions(users_df, articles)

    # train personalization model
    personalizer = PersonalizedRanker()
    if len(interactions_df) > 0:
        personalizer.fit_collaborative_filtering(interactions_df)

    logger.info("✅ User behavior simulation complete")
    return users_df, interactions_df, personalizer


def run_evaluation(config, logger, search_engine, l2r_system, evaluator):
    """run comprehensive evaluation and a/b testing"""
    logger.info("📊 Running evaluation and A/B testing...")

    # initialize a/b testing framework
    ab_framework = ABTestFramework(evaluator)

    # run a/b tests comparing different methods
    experiments = {}

    # test 1: bm25 vs hybrid
    if hasattr(search_engine, 'bm25_search') and hasattr(search_engine, 'hybrid_search'):
        exp1 = ab_framework.run_ab_test(
            control_method='bm25',
            treatment_method='hybrid',
            test_name='bm25_vs_hybrid',
            k=5
        )
        experiments['bm25_vs_hybrid'] = exp1

    # test 2: hybrid vs ml-enhanced (if l2r trained)
    if l2r_system is not None:
        # create a wrapper for ml-enhanced search
        def ml_search(query, method='ml_enhanced', k=10):
            return l2r_system.ml_enhanced_search(query, k)

        # temporarily add ml search method
        original_search = search_engine.search

        def enhanced_search(query, method='hybrid', k=10):
            if method == 'ml_enhanced':
                return ml_search(query, k=k)
            else:
                return original_search(query, method, k)

        search_engine.search = enhanced_search

        exp2 = ab_framework.run_ab_test(
            control_method='hybrid',
            treatment_method='ml_enhanced',
            test_name='hybrid_vs_ml',
            k=5
        )
        experiments['hybrid_vs_ml'] = exp2

        # restore original search method
        search_engine.search = original_search

    # generate final report
    final_report = ab_framework.generate_experiment_report()

    logger.info("✅ Evaluation complete")
    return experiments, final_report


def demonstrate_system(config, logger, search_engine, l2r_system, personalizer, users_df):
    """demonstrate the complete system capabilities"""
    logger.info("🎯 Demonstrating system capabilities...")

    test_queries = [
        "artificial intelligence machine learning",
        "climate change environment",
        "stock market investment",
        "health and medicine",
        "sports news football"
    ]

    print("\n" + "=" * 80)
    print("🎪 MULTIMODAL SEARCH ENGINE DEMONSTRATION")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Demo {i}: '{query}'")
        print("-" * 50)

        # basic hybrid search
        print("📊 Hybrid Search Results:")
        try:
            if hasattr(search_engine, 'hybrid_search'):
                results = search_engine.hybrid_search(query, k=3)
            else:
                results = search_engine.distributed_search(query, top_k=3)

            for j, result in enumerate(results[:3], 1):
                print(f"   {j}. {result['title'][:60]}...")
                print(
                    f"      Category: {result.get('category', 'N/A')} | Score: {result.get('score', result.get('hybrid_score', 0)):.4f}")
        except Exception as e:
            print(f"   ❌ Error in hybrid search: {e}")

        # ml-enhanced search (if available)
        if l2r_system is not None:
            print("\n🤖 ML-Enhanced Search Results:")
            try:
                ml_results = l2r_system.ml_enhanced_search(query, k=3)
                for j, result in enumerate(ml_results[:3], 1):
                    print(f"   {j}. {result['title'][:60]}...")
                    print(
                        f"      Category: {result['category']} | ML Score: {result['ml_score']:.4f} | Final: {result['final_score']:.4f}")
            except Exception as e:
                print(f"   ❌ Error in ML search: {e}")

        # personalized search (if available)
        if personalizer is not None and len(users_df) > 0:
            print("\n👤 Personalized Search Results:")
            try:
                personalized_engine = PersonalizedSearchEngine(search_engine, personalizer)
                test_user = users_df['user_id'].iloc[0]
                personal_results = personalized_engine.personalized_search(query, test_user, k=3)
            except Exception as e:
                print(f"   ❌ Error in personalized search: {e}")

        if i < len(test_queries):
            time.sleep(2)  # brief pause between demos

    print(f"\n🎉 System demonstration complete!")
    logger.info("✅ System demonstration complete")


def main():
    """main pipeline orchestrator"""
    parser = argparse.ArgumentParser(description='Multimodal Search Engine Pipeline')
    parser.add_argument('--stage',
                        choices=['collect', 'index', 'train', 'evaluate', 'simulate', 'demo', 'all'],
                        default='all',
                        help='Which stage to run')
    parser.add_argument('--articles-file',
                        help='Path to existing articles file (skip collection)')
    parser.add_argument('--target-articles', type=int,
                        help='Number of articles to collect')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # setup configuration
    config = Config()
    if args.target_articles:
        config.TARGET_ARTICLES = args.target_articles

    # setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # print startup banner
    print("\n" + "=" * 80)
    print("🚀 MULTIMODAL SEARCH ENGINE WITH LEARNING-TO-RANK")
    print("=" * 80)
    print(f"📊 Target articles: {config.TARGET_ARTICLES}")
    print(f"🎯 Stage: {args.stage}")
    print(f"📝 Logs: logs/search_engine.log")
    print("=" * 80)

    # pipeline variables
    articles_file = args.articles_file
    articles = None
    search_engine = None
    l2r_system = None
    evaluator = None
    users_df = None
    interactions_df = None
    personalizer = None

    try:
        # stage 1: data collection
        if args.stage in ['collect', 'all'] and not articles_file:
            articles_file, articles = collect_data(config, logger)

        # load existing articles if file provided
        if articles_file and not articles:
            logger.info(f"📚 Loading articles from {articles_file}")
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            logger.info(f"✅ Loaded {len(articles)} articles")

        # stage 2: build search indices
        if args.stage in ['index', 'train', 'evaluate', 'demo', 'all']:
            if not articles_file:
                logger.error("❌ No articles file available for indexing")
                return 1
            search_engine = build_search_indices(config, logger, articles_file)

        # stage 3: train ranking models
        if args.stage in ['train', 'evaluate', 'demo', 'all']:
            l2r_system, evaluator, model_performance = train_ranking_models(config, logger, search_engine)

        # stage 4: simulate user behavior
        if args.stage in ['simulate', 'demo', 'all']:
            if articles:
                users_df, interactions_df, personalizer = simulate_user_behavior(config, logger, articles)

        # stage 5: evaluation and a/b testing
        if args.stage in ['evaluate', 'all']:
            if evaluator:
                experiments, final_report = run_evaluation(config, logger, search_engine, l2r_system, evaluator)

        # stage 6: system demonstration
        if args.stage in ['demo', 'all']:
            demonstrate_system(config, logger, search_engine, l2r_system, personalizer, users_df)

        # final summary
        print("\n" + "=" * 80)
        print("🎉 PIPELINE EXECUTION COMPLETE")
        print("=" * 80)

        if articles:
            print(f"📊 Articles processed: {len(articles)}")
        if l2r_system and hasattr(l2r_system, 'training_data') and l2r_system.training_data is not None:
            print(f"🤖 Training samples generated: {len(l2r_system.training_data)}")
        if users_df is not None:
            print(f"👥 User profiles simulated: {len(users_df)}")
        if interactions_df is not None:
            print(f"🔍 User interactions simulated: {len(interactions_df)}")

        print(f"📝 Detailed logs saved to: logs/search_engine.log")
        print(f"🎯 System ready for production deployment!")
        print("=" * 80)

        logger.info("🎉 Pipeline execution completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("⚠️  Pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        print(f"\n❌ Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)