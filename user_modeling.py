# user_modeling.py - 用戶建模模塊
# user behavior simulation and personalized ranking system

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import random
from datetime import datetime, timedelta
from collections import defaultdict
from config import Config


class UserBehaviorSimulator:
    """
    simulate realistic user behavior patterns for search evaluation
    demonstrates: user modeling, behavioral data generation, engagement patterns
    """

    def __init__(self, num_users=1000, num_articles=10000):
        self.num_users = num_users
        self.num_articles = num_articles

        # user segments with different behavior patterns
        self.user_segments = {
            'tech_enthusiast': 0.25,
            'news_consumer': 0.30,
            'health_focused': 0.20,
            'business_oriented': 0.15,
            'general_browser': 0.10
        }

        print(f"👥 initializing user behavior simulator")
        print(f"   users: {num_users}, articles: {num_articles}")

    def generate_user_profiles(self) -> pd.DataFrame:
        """generate diverse user profiles with preferences"""
        print("👤 generating user profiles...")

        users = []
        for user_id in range(self.num_users):
            # assign user to segment
            segment = np.random.choice(
                list(self.user_segments.keys()),
                p=list(self.user_segments.values())
            )

            # generate user characteristics
            user_profile = {
                'user_id': user_id,
                'segment': segment,
                'tech_affinity': self._get_affinity('technology', segment),
                'news_affinity': self._get_affinity('news', segment),
                'health_affinity': self._get_affinity('health', segment),
                'business_affinity': self._get_affinity('business', segment),
                'sports_affinity': self._get_affinity('sports', segment),
                'avg_session_length': np.random.normal(15, 5),
                'daily_searches': np.random.poisson(8),
                'click_propensity': np.random.beta(2, 5),
                'dwell_time_preference': np.random.gamma(2, 2),
            }

            users.append(user_profile)

        user_df = pd.DataFrame(users)
        print(f"✅ generated {len(user_df)} user profiles")
        return user_df

    def _get_affinity(self, category: str, segment: str) -> float:
        """calculate user affinity for content category based on segment"""
        base_affinity = 0.3

        segment_boosts = {
            'tech_enthusiast': {'technology': 0.6, 'news': 0.2, 'health': 0.1, 'business': 0.3, 'sports': 0.1},
            'news_consumer': {'technology': 0.2, 'news': 0.7, 'health': 0.3, 'business': 0.4, 'sports': 0.3},
            'health_focused': {'technology': 0.1, 'news': 0.2, 'health': 0.6, 'business': 0.1, 'sports': 0.4},
            'business_oriented': {'technology': 0.3, 'news': 0.4, 'health': 0.1, 'business': 0.6, 'sports': 0.2},
            'general_browser': {'technology': 0.2, 'news': 0.2, 'health': 0.2, 'business': 0.2, 'sports': 0.2}
        }

        boost = segment_boosts.get(segment, {}).get(category, 0)
        return min(base_affinity + boost + np.random.normal(0, 0.1), 1.0)

    def simulate_search_interactions(self, users_df: pd.DataFrame, articles: list) -> pd.DataFrame:
        """simulate realistic search and interaction patterns"""
        print("🔍 simulating search interactions...")

        interactions = []

        for _, user in users_df.iterrows():
            num_searches = int(user['daily_searches'] * 7)  # week of activity

            for search_idx in range(num_searches):
                # generate search query based on user preferences
                query, query_category = self._generate_user_query(user)

                # simulate search results
                search_results = self._get_relevant_articles(articles, query_category, 10)

                # simulate user interactions
                for rank, article in enumerate(search_results):
                    interaction = self._simulate_interaction(user, article, rank, query)
                    if interaction:
                        interactions.append(interaction)

        interactions_df = pd.DataFrame(interactions)
        print(f"✅ simulated {len(interactions_df)} user interactions")
        return interactions_df

    def _generate_user_query(self, user: pd.Series) -> tuple:
        """generate search query based on user preferences"""
        categories = ['technology', 'news', 'health', 'business', 'sports']

        # safely get affinities with fallback values
        affinities = []
        for cat in categories:
            affinity_key = f'{cat}_affinity'
            affinity = user.get(affinity_key, 0.2)  # default affinity
            affinities.append(affinity)

        # normalize affinities
        total_affinity = sum(affinities)
        if total_affinity > 0:
            affinities = [a / total_affinity for a in affinities]
        else:
            affinities = [0.2] * len(categories)  # uniform distribution

        chosen_category = np.random.choice(categories, p=affinities)

        query_templates = {
            'technology': ['machine learning', 'artificial intelligence', 'programming', 'software development',
                           'tech news'],
            'news': ['breaking news', 'current events', 'politics', 'world news', 'latest updates'],
            'health': ['health tips', 'medical research', 'fitness', 'nutrition', 'mental health'],
            'business': ['stock market', 'investment', 'business news', 'economics', 'finance'],
            'sports': ['sports news', 'football', 'basketball', 'soccer', 'sports scores']
        }

        query = np.random.choice(query_templates[chosen_category])
        return query, chosen_category

    def _get_relevant_articles(self, articles: list, category: str, limit: int) -> list:
        """get articles relevant to user query category"""
        category_articles = [a for a in articles if a.get('category') == category]
        other_articles = [a for a in articles if a.get('category') != category]

        selected = (category_articles[:limit // 2] +
                    random.sample(other_articles, min(limit // 2, len(other_articles))))

        return selected[:limit]

    def _simulate_interaction(self, user: pd.Series, article: dict, rank: int, query: str) -> dict:
        """simulate user interaction with search result"""
        # click probability calculation
        position_bias = 1.0 / (rank + 1) ** 0.5

        # safely get category affinity
        article_category = article.get('category', 'general')
        affinity_key = f"{article_category}_affinity"
        content_match = user.get(affinity_key, 0.3)  # default match score

        click_prob = user.get('click_propensity', 0.5) * position_bias * content_match

        if np.random.random() < click_prob:
            # simulate engagement
            base_dwell = user.get('dwell_time_preference', 2.0) * 30
            article_quality_boost = article.get('score', 0) / 100
            dwell_time = max(5, base_dwell + article_quality_boost * 60)

            engagement_score = min(dwell_time / 120, 1.0)

            return {
                'user_id': user['user_id'],
                'article_id': article['id'],
                'query': query,
                'rank': rank,
                'clicked': 1,
                'dwell_time': dwell_time,
                'engagement_score': engagement_score,
                'article_category': article.get('category'),
                'user_segment': user['segment'],
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30))
            }

        return None


class PersonalizedRanker:
    """
    personalized ranking system using collaborative filtering
    demonstrates: personalization, collaborative filtering, ranking optimization
    """

    def __init__(self):
        self.user_article_matrix = None
        self.user_embeddings = None
        self.article_embeddings = None
        self.svd_model = None
        print("🤖 personalized ranker initialized")

    def fit_collaborative_filtering(self, interactions_df: pd.DataFrame):
        """train collaborative filtering model for personalization"""
        print("📊 training collaborative filtering model...")

        # create user-article interaction matrix
        pivot_data = interactions_df.pivot_table(
            index='user_id',
            columns='article_id',
            values='engagement_score',
            fill_value=0
        )

        self.user_article_matrix = pivot_data.values
        self.user_ids = pivot_data.index
        self.article_ids = pivot_data.columns

        # matrix factorization using SVD
        # adjust components based on available features
        n_components = min(50, min(self.user_article_matrix.shape) - 1)
        n_components = max(1, n_components)  # at least 1 component

        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_embeddings = self.svd_model.fit_transform(self.user_article_matrix)
        self.article_embeddings = self.svd_model.components_.T

        print(f"✅ collaborative filtering trained")
        print(f"   user embeddings: {self.user_embeddings.shape}")
        print(f"   article embeddings: {self.article_embeddings.shape}")

    def get_personalized_scores(self, user_id: int, candidate_articles: list) -> dict:
        """generate personalized relevance scores for candidate articles"""
        if self.user_embeddings is None:
            return {}

        try:
            # get user embedding
            user_idx = list(self.user_ids).index(user_id)
            user_embedding = self.user_embeddings[user_idx]

            # calculate personalized scores
            personalized_scores = {}

            for article in candidate_articles:
                article_id = article['id']

                if article_id in self.article_ids:
                    # get article embedding
                    article_idx = list(self.article_ids).index(article_id)
                    article_embedding = self.article_embeddings[article_idx]

                    # compute personalized score
                    personal_score = np.dot(user_embedding, article_embedding)
                    personalized_scores[article_id] = personal_score
                else:
                    # cold start: use average score
                    personalized_scores[article_id] = np.mean(user_embedding)

            return personalized_scores

        except (ValueError, IndexError):
            # user not in training data
            return {article['id']: 0.5 for article in candidate_articles}

    def extract_personalization_features(self, user_id: int, article_id: int, interactions_df: pd.DataFrame) -> dict:
        """extract user-specific features for ranking model"""
        user_history = interactions_df[interactions_df['user_id'] == user_id]

        if len(user_history) == 0:
            return {
                'user_click_history': 0,
                'user_avg_dwell': 30.0,
                'user_category_preference': 0.5,
                'user_engagement_rate': 0.5
            }

        # user behavior features
        features = {
            'user_click_history': len(user_history),
            'user_avg_dwell': user_history['dwell_time'].mean(),
            'user_engagement_rate': user_history['engagement_score'].mean(),
        }

        # category-specific features
        article_category = interactions_df[interactions_df['article_id'] == article_id]['article_category'].iloc[
            0] if len(interactions_df[interactions_df['article_id'] == article_id]) > 0 else 'general'

        category_history = user_history[user_history['article_category'] == article_category]
        features['user_category_preference'] = len(category_history) / len(user_history) if len(
            user_history) > 0 else 0.5
        features['category_avg_engagement'] = category_history['engagement_score'].mean() if len(
            category_history) > 0 else 0.5

        return features


class PersonalizedSearchEngine:
    """
    search engine with personalized ranking
    demonstrates: personalization integration, hybrid ranking
    """

    def __init__(self, base_search_engine, personalizer):
        self.base_engine = base_search_engine
        self.personalizer = personalizer
        print("🎯 personalized search engine ready")

    def personalized_search(self, query: str, user_id: int, k: int = 10) -> list:
        """search with personalized reranking"""
        # get base search results
        base_results = self.base_engine.search(query, method='hybrid', k=k * 2)

        # get personalized scores
        personal_scores = self.personalizer.get_personalized_scores(user_id, base_results)

        # combine base relevance with personalization
        for result in base_results:
            base_score = result.get('hybrid_score', 0.5)
            personal_score = personal_scores.get(result['article_id'], 0.5)

            # weighted combination
            result['personalized_score'] = 0.7 * base_score + 0.3 * personal_score
            result['personal_component'] = personal_score

        # rerank by personalized scores
        base_results.sort(key=lambda x: x['personalized_score'], reverse=True)

        print(f"🎯 personalized search for user {user_id}: '{query}'")
        print(f"📊 personalized results:")
        for i, result in enumerate(base_results[:5], 1):
            print(f"   {i}. {result['title'][:60]}...")
            print(f"      category: {result['category']} | personalized_score: {result['personalized_score']:.4f}")

        return base_results[:k]


class CTRPredictor:
    """
    click-through rate prediction model
    demonstrates: engagement prediction, user behavior modeling
    """

    def __init__(self):
        self.ctr_model = None
        self.feature_scaler = StandardScaler()
        print("📈 ctr predictor initialized")

    def extract_ctr_features(self, user_profile, article, query, position):
        """extract features for ctr prediction"""
        features = {
            # user features
            'user_click_propensity': user_profile.get('click_propensity', 0.5),
            'user_session_length': user_profile.get('avg_session_length', 15),
            'user_daily_searches': user_profile.get('daily_searches', 8),

            # article features
            'article_score': article.get('score', 0),
            'article_category_match': 1 if article.get('category') in query.lower() else 0,
            'title_length': len(article.get('title', '')),
            'content_length': len(article.get('content', '')),

            # context features
            'position': position,
            'position_bias': 1.0 / (position + 1) ** 0.5,
            'query_length': len(query.split()),

            # interaction features
            'title_query_overlap': self._compute_overlap(article.get('title', ''), query),
        }

        return list(features.values())

    def _compute_overlap(self, text1, text2):
        """compute word overlap between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words2:
            return 0

        return len(words1 & words2) / len(words2)

    def predict_ctr(self, user_profile, articles, query):
        """predict click-through rates for search results"""
        if self.ctr_model is None:
            # simple heuristic model if no trained model
            return self._heuristic_ctr_prediction(user_profile, articles, query)

        # use trained model for prediction
        features = []
        for position, article in enumerate(articles):
            article_features = self.extract_ctr_features(user_profile, article, query, position)
            features.append(article_features)

        features_scaled = self.feature_scaler.transform(features)
        ctr_predictions = self.ctr_model.predict_proba(features_scaled)[:, 1]

        return ctr_predictions

    def _heuristic_ctr_prediction(self, user_profile, articles, query):
        """heuristic ctr prediction when no model is trained"""
        ctr_predictions = []

        for position, article in enumerate(articles):
            # position bias
            position_bias = 1.0 / (position + 1) ** 0.5

            # relevance estimate
            title_match = self._compute_overlap(article.get('title', ''), query)
            category_bonus = 0.2 if article.get('category', '').lower() in query.lower() else 0

            # user propensity
            user_propensity = user_profile.get('click_propensity', 0.5)

            # combine factors
            ctr = position_bias * (0.3 + 0.4 * title_match + category_bonus) * user_propensity
            ctr_predictions.append(min(ctr, 0.9))  # cap at 90%

        return ctr_predictions


# standalone execution for testing
if __name__ == "__main__":
    config = Config()

    print("🧪 testing user modeling system...")

    # test user behavior simulator
    simulator = UserBehaviorSimulator(num_users=10, num_articles=100)
    users_df = simulator.generate_user_profiles()

    # create sample articles for testing
    sample_articles = [
        {'id': i, 'title': f'Article {i}', 'category': ['technology', 'business', 'health', 'sports', 'science'][i % 5],
         'score': np.random.randint(1, 100)}
        for i in range(20)
    ]

    interactions_df = simulator.simulate_search_interactions(users_df, sample_articles)

    print(f"📊 simulation results:")
    print(f"   users: {len(users_df)}")
    print(f"   interactions: {len(interactions_df)}")
    print(f"   average engagement: {interactions_df['engagement_score'].mean():.3f}")

    # test collaborative filtering
    personalizer = PersonalizedRanker()
    if len(interactions_df) > 0:
        personalizer.fit_collaborative_filtering(interactions_df)

        # test personalized scores
        test_user = users_df['user_id'].iloc[0]
        personal_scores = personalizer.get_personalized_scores(test_user, sample_articles[:5])
        print(f"🎯 personalized scores for user {test_user}: {personal_scores}")

    print("✅ user modeling system test complete")