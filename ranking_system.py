# ranking_system.py - 排序系統模塊
# learning to rank system with comprehensive feature engineering

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import json
import random
import warnings
from openai import OpenAI
from config import Config

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning)


class RankingFeatureExtractor:
    """
    comprehensive feature engineering for learning to rank
    demonstrates: feature engineering skills, ranking signals
    """

    def __init__(self, search_engine, evaluator):
        self.search_engine = search_engine
        self.evaluator = evaluator
        self.article_stats = self._precompute_article_stats()
        print("🔧 ranking feature extractor initialized")

    def _precompute_article_stats(self):
        """precompute article-level statistics"""
        stats = {}

        for i, article in enumerate(self.search_engine.articles):
            content = f"{article['title']} {article.get('content', '')}"

            stats[i] = {
                'content_length': len(content),
                'title_length': len(article['title']),
                'word_count': len(content.split()),
                'avg_word_length': np.mean([len(word) for word in content.split()]) if content.split() else 0,
                'exclamation_count': content.count('!'),
                'question_count': content.count('?'),
                'capital_ratio': sum(1 for c in content if c.isupper()) / len(content) if content else 0,
                'category': article['category'],
                'source': article['source']
            }

        return stats

    def extract_query_features(self, query):
        """extract query-level features"""
        query_words = query.lower().split()

        return {
            'query_length': len(query),
            'query_word_count': len(query_words),
            'avg_query_word_length': np.mean([len(word) for word in query_words]) if query_words else 0,
            'has_question_words': any(word in query.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']),
            'has_tech_terms': any(
                word in query.lower() for word in ['ai', 'technology', 'computer', 'digital', 'tech']),
            'has_business_terms': any(
                word in query.lower() for word in ['stock', 'market', 'investment', 'finance', 'business']),
            'has_health_terms': any(
                word in query.lower() for word in ['health', 'medicine', 'medical', 'covid', 'disease']),
            'has_sports_terms': any(
                word in query.lower() for word in ['sports', 'football', 'soccer', 'basketball', 'game'])
        }

    def extract_relevance_features(self, query, article_id, search_results=None):
        """extract query-article relevance features"""
        article = self.search_engine.articles[article_id]
        content = f"{article['title']} {article.get('content', '')}".lower()
        query_lower = query.lower()
        query_words = query_lower.split()

        # text matching features
        title_matches = sum(1 for word in query_words if word in article['title'].lower())
        content_matches = sum(1 for word in query_words if word in content)
        exact_phrase_match = 1 if query_lower in content else 0

        # tf-idf and bm25 scores
        tfidf_score = 0
        bm25_score = 0
        if search_results:
            for result in search_results:
                if result.get('article_id') == article_id:
                    tfidf_score = float(result.get('tfidf_score', 0))
                    bm25_score = float(result.get('bm25_score', 0))
                    break

        # category matching
        query_category_hints = {
            'technology': ['tech', 'ai', 'computer', 'digital', 'software'],
            'business': ['business', 'stock', 'market', 'finance', 'investment'],
            'health': ['health', 'medical', 'covid', 'medicine', 'disease'],
            'sports': ['sports', 'football', 'soccer', 'basketball', 'game'],
            'science': ['science', 'research', 'study', 'climate', 'environment']
        }

        category_match = 0
        for category, keywords in query_category_hints.items():
            if article['category'] == category and any(keyword in query_lower for keyword in keywords):
                category_match = 1
                break

        # safe division calculations
        query_coverage = 0
        if query_words:
            query_coverage = (title_matches + content_matches) / len(query_words)

        title_query_overlap = 0
        if query_words:
            query_set = set(query_words)
            title_set = set(article['title'].lower().split())
            if query_set:
                title_query_overlap = len(query_set & title_set) / len(query_set)

        return {
            'title_word_matches': int(title_matches),
            'content_word_matches': int(content_matches),
            'query_coverage': float(query_coverage),
            'exact_phrase_match': int(exact_phrase_match),
            'tfidf_score': float(tfidf_score),
            'bm25_score': float(bm25_score),
            'category_match': int(category_match),
            'title_query_overlap': float(title_query_overlap)
        }

    def extract_quality_features(self, article_id):
        """extract article quality features"""
        article = self.search_engine.articles[article_id]
        stats = self.article_stats[article_id]

        # source authority scoring
        source_authority = {
            'BBC News': 0.95, 'Reuters': 0.95, 'Bloomberg': 0.90,
            'NBC News': 0.85, 'Fortune': 0.80, 'TechCrunch': 0.75,
            'MarketWatch': 0.70, 'Yahoo Entertainment': 0.60
        }.get(article['source'], 0.50)

        # content quality indicators
        has_image = 1 if article.get('image_url') else 0
        content_completeness = min(stats['content_length'] / 500, 1.0)
        title_quality = 1 if 10 <= stats['title_length'] <= 100 else 0

        return {
            'source_authority': source_authority,
            'has_image': has_image,
            'content_length_norm': content_completeness,
            'title_quality': title_quality,
            'word_count': stats['word_count'],
            'avg_word_length': stats['avg_word_length'],
            'exclamation_ratio': stats['exclamation_count'] / stats['content_length'] if stats[
                                                                                             'content_length'] > 0 else 0,
            'capital_ratio': stats['capital_ratio']
        }

    def extract_freshness_features(self, article_id):
        """extract temporal/freshness features"""
        article = self.search_engine.articles[article_id]

        try:
            pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            now = datetime.now()
            hours_old = (now - pub_date).total_seconds() / 3600

            freshness_score = np.exp(-hours_old / 24)

            return {
                'hours_since_published': hours_old,
                'freshness_score': freshness_score,
                'is_recent': 1 if hours_old <= 24 else 0,
                'is_very_recent': 1 if hours_old <= 6 else 0
            }
        except:
            return {
                'hours_since_published': 9999,
                'freshness_score': 0,
                'is_recent': 0,
                'is_very_recent': 0
            }

    def extract_all_features(self, query, article_id, search_results=None):
        """combine all feature types into comprehensive feature vector"""
        features = {}

        try:
            features.update(self.extract_query_features(query))
            features.update(self.extract_relevance_features(query, article_id, search_results))
            features.update(self.extract_quality_features(article_id))
            features.update(self.extract_freshness_features(article_id))

            # position features
            if search_results:
                position = -1
                for i, r in enumerate(search_results):
                    if r.get('article_id') == article_id:
                        position = i
                        break

                features.update({
                    'initial_rank': position if position >= 0 else 999,
                    'in_top_3': 1 if 0 <= position < 3 else 0,
                    'in_top_5': 1 if 0 <= position < 5 else 0
                })
            else:
                features.update({
                    'initial_rank': 999,
                    'in_top_3': 0,
                    'in_top_5': 0
                })

            # ensure all values are numeric
            for key, value in features.items():
                if isinstance(value, (list, dict, set)):
                    features[key] = str(value)
                elif not isinstance(value, (int, float, bool)):
                    features[key] = 0

        except Exception as e:
            print(f"❌ error extracting features for article {article_id}: {e}")
            features = {
                'query_length': len(query),
                'article_id_num': article_id,
                'basic_feature': 1
            }

        return features


class LearningToRankSystem:
    """
    complete learning to rank system
    demonstrates: ml model development, training, evaluation
    """

    def __init__(self, search_engine, evaluator):
        self.search_engine = search_engine
        self.evaluator = evaluator
        self.feature_extractor = RankingFeatureExtractor(search_engine, evaluator)

        self.models = {}
        self.training_data = None
        self.best_model = None
        self.feature_names = None

        print("🤖 learning to rank system initialized")

    def generate_training_data(self, num_samples=500):
        """generate training data for learning to rank"""
        print("📊 generating training data...")

        training_samples = []
        queries = [qd['query'] for qd in self.evaluator.evaluation_queries]

        # expand query set
        expanded_queries = queries + [
            "breaking news today", "latest technology", "health tips",
            "financial markets", "sports updates", "business news",
            "science research", "investment advice", "medical breakthrough"
        ]

        for query in expanded_queries:
            try:
                print(f"   processing query: '{query}'")

                # get search results
                bm25_results = self.search_engine.bm25_search(query, k=15)
                tfidf_results = self.search_engine.tfidf_search(query, k=15)

                # combine results
                all_article_ids = set()
                all_results = []

                for results in [bm25_results, tfidf_results]:
                    for result in results:
                        aid = result.get('article_id')
                        if aid is not None and aid not in all_article_ids:
                            all_article_ids.add(aid)
                            all_results.append(result)

                # add random articles
                available_articles = list(range(len(self.search_engine.articles)))
                random_articles = random.sample(available_articles,
                                                min(5, len(available_articles)))
                for aid in random_articles:
                    if aid not in all_article_ids:
                        all_results.append({'article_id': aid})

                # extract features
                for result in all_results[:20]:
                    try:
                        article_id = result.get('article_id')
                        if article_id is None:
                            continue

                        features = self.feature_extractor.extract_all_features(
                            query, article_id, all_results
                        )

                        # generate relevance label
                        if query in self.evaluator.ground_truth:
                            relevance = self.evaluator.ground_truth[query].get(article_id, 0)
                        else:
                            relevance = self._estimate_relevance(query, article_id)

                        sample = {
                            'query': query,
                            'article_id': article_id,
                            'relevance': relevance,
                            **features
                        }

                        training_samples.append(sample)

                    except Exception as e:
                        print(f"      ⚠️  error processing article {result.get('article_id')}: {e}")
                        continue

            except Exception as e:
                print(f"   ❌ error processing query '{query}': {e}")
                continue

        if training_samples:
            self.training_data = pd.DataFrame(training_samples)
            print(f"✅ generated {len(training_samples)} training samples")

            feature_cols = [col for col in self.training_data.columns
                            if col not in ['query', 'article_id', 'relevance']]
            print(f"📊 feature count: {len(feature_cols)}")
        else:
            print("❌ no training samples generated")

        return self.training_data

    def _estimate_relevance(self, query, article_id):
        """estimate relevance for expanded queries"""
        article = self.search_engine.articles[article_id]
        query_lower = query.lower()
        content_lower = f"{article['title']} {article.get('content', '')}".lower()

        score = 0
        query_words = query_lower.split()

        # keyword matching
        matches = sum(1 for word in query_words if word in content_lower)
        score += matches

        # category relevance
        category_relevance = {
            ('technology', ['tech', 'ai', 'computer']): 2,
            ('business', ['business', 'finance', 'market']): 2,
            ('health', ['health', 'medical']): 2,
            ('sports', ['sports', 'football']): 2,
            ('science', ['science', 'research']): 2
        }

        for (category, keywords), bonus in category_relevance.items():
            if article['category'] == category and any(kw in query_lower for kw in keywords):
                score += bonus

        return min(score, 3)

    def train_models(self):
        """train multiple ranking models and compare performance"""
        if self.training_data is None:
            print("❌ no training data available")
            return

        print("🤖 training ranking models...")

        feature_cols = [col for col in self.training_data.columns
                        if col not in ['query', 'article_id', 'relevance']]

        X = self.training_data[feature_cols].fillna(0)
        y = self.training_data['relevance']

        print(f"📊 training on {len(X)} samples with {len(feature_cols)} features")

        models_to_train = {
            'xgboost': xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_performance = {}

        for name, model in models_to_train.items():
            print(f"   🔧 training {name}...")

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

            model_performance[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"      test r²: {test_r2:.4f}")
            print(f"      cv r²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        best_model_name = max(model_performance.keys(),
                              key=lambda x: model_performance[x]['cv_mean'])

        self.models = model_performance
        self.best_model = model_performance[best_model_name]['model']
        self.feature_names = feature_cols

        print(f"\n🏆 best model: {best_model_name}")
        print(f"   cv r²: {model_performance[best_model_name]['cv_mean']:.4f}")

        self._analyze_feature_importance(best_model_name)

        return model_performance

    def _analyze_feature_importance(self, model_name):
        """analyze feature importance for interpretability"""
        model = self.models[model_name]['model']

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print(f"\n🔍 top feature importance ({model_name}):")
            for _, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")

    def ml_enhanced_search(self, query, k=10):
        """search using ml-enhanced ranking"""
        print(f"🤖 ml-enhanced search for: '{query}'")

        if not hasattr(self, 'best_model') or self.best_model is None:
            print("❌ no trained model available")
            return []

        initial_results = self.search_engine.hybrid_search(query, k=k * 3)

        if not initial_results:
            return []

        ml_results = []

        for result in initial_results:
            article_id = result['article_id']

            features = self.feature_extractor.extract_all_features(query, article_id, initial_results)

            # predict relevance score
            feature_vector = [features.get(fname, 0) for fname in self.feature_names]
            feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
            ml_score = self.best_model.predict(feature_df)[0]

            result_copy = result.copy()
            result_copy['ml_score'] = ml_score
            result_copy['final_score'] = 0.7 * ml_score + 0.3 * result.get('hybrid_score', 0)

            ml_results.append(result_copy)

        ml_results.sort(key=lambda x: x['final_score'], reverse=True)

        print(f"📊 ml-enhanced results:")
        for i, result in enumerate(ml_results[:5], 1):
            print(f"   {i}. {result['title'][:60]}...")
            print(
                f"      category: {result['category']} | ml_score: {result['ml_score']:.4f} | final: {result['final_score']:.4f}")

        return ml_results[:k]


# standalone execution for testing
if __name__ == "__main__":
    config = Config()

    print("🧪 testing ranking system...")
    print("   (requires search engine and evaluator to be initialized)")
    print("   run main.py for complete pipeline test")

    print("✅ ranking system module loaded successfully")