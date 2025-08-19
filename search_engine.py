# search_engine.py - 搜索引擎模塊
# scalable search engine with multiple retrieval strategies

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from collections import Counter, defaultdict
import pickle
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import psutil
from openai import OpenAI
from config import Config
import warnings

warnings.filterwarnings('ignore')


class MultiModalSearchEngine:
    """
    basic multimodal search engine with hybrid ranking
    demonstrates: multiple retrieval algorithms, semantic search
    """

    def __init__(self, openai_api_key):
        self.articles = []
        self.openai_client = OpenAI(api_key=openai_api_key)

        # search indices
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.article_embeddings = {}

        # evaluation data
        self.search_logs = []

        print("🔍 basic search engine initialized")

    def load_articles(self, articles_file):
        """load articles and build search indices"""
        print(f"📚 loading articles from {articles_file}...")

        with open(articles_file, 'r', encoding='utf-8') as f:
            self.articles = json.load(f)

        print(f"✅ loaded {len(self.articles)} articles")

        # prepare text data for indexing
        self.article_texts = []
        for article in self.articles:
            text = f"{article['title']} {article.get('description', '')} {article.get('content', '')}"
            self.article_texts.append(text.lower())

        print("🔧 building search indices...")
        self._build_tfidf_index()
        print("✅ basic search engine ready!")

    def _build_tfidf_index(self):
        """build tf-idf index for keyword matching"""
        print("   🔧 building tf-idf index...")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.article_texts)
        print(f"   ✅ tf-idf index built: {self.tfidf_matrix.shape[1]} features")

    def bm25_search(self, query, k=10):
        """bm25 ranking algorithm"""
        query_terms = query.lower().split()
        scores = np.zeros(len(self.articles))

        # bm25 parameters
        k1, b = 1.2, 0.75
        avg_doc_len = np.mean([len(text.split()) for text in self.article_texts])

        for term in query_terms:
            df = sum(1 for text in self.article_texts if term in text)
            if df == 0:
                continue

            idf = math.log((len(self.articles) - df + 0.5) / (df + 0.5))

            for i, text in enumerate(self.article_texts):
                tf = text.count(term)
                doc_len = len(text.split())

                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
                scores[i] += score

        top_indices = np.argsort(scores)[::-1][:k]
        results = []

        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'article_id': int(idx),
                    'title': self.articles[idx]['title'],
                    'category': self.articles[idx]['category'],
                    'source': self.articles[idx]['source'],
                    'url': self.articles[idx]['url'],
                    'bm25_score': float(scores[idx])
                })

        return results

    def tfidf_search(self, query, k=10):
        """tf-idf cosine similarity search"""
        query_vector = self.tfidf_vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:k]
        results = []

        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'article_id': int(idx),
                    'title': self.articles[idx]['title'],
                    'category': self.articles[idx]['category'],
                    'source': self.articles[idx]['source'],
                    'url': self.articles[idx]['url'],
                    'tfidf_score': float(similarities[idx])
                })

        return results

    def semantic_search(self, query, k=10):
        """semantic search using openai embeddings"""
        print(f"🤖 generating semantic embedding for: '{query}'")

        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = np.array(response.data[0].embedding)

            similarities = []
            for i, article in enumerate(self.articles):
                if f'embedding_{i}' not in self.article_embeddings:
                    text = f"{article['title']} {article.get('description', '')}"
                    article_response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    self.article_embeddings[f'embedding_{i}'] = np.array(article_response.data[0].embedding)

                article_embedding = self.article_embeddings[f'embedding_{i}']
                similarity = np.dot(query_embedding, article_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding)
                )
                similarities.append(similarity)

            similarities = np.array(similarities)
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                results.append({
                    'article_id': int(idx),
                    'title': self.articles[idx]['title'],
                    'category': self.articles[idx]['category'],
                    'source': self.articles[idx]['source'],
                    'url': self.articles[idx]['url'],
                    'semantic_score': float(similarities[idx])
                })

            return results

        except Exception as e:
            print(f"❌ semantic search failed: {e}")
            return []

    def hybrid_search(self, query, k=10, weights={'bm25': 0.4, 'tfidf': 0.3, 'semantic': 0.3}):
        """hybrid search combining multiple algorithms"""
        print(f"🔍 hybrid search for: '{query}'")

        bm25_results = self.bm25_search(query, k * 2)
        tfidf_results = self.tfidf_search(query, k * 2)
        semantic_results = self.semantic_search(query, k * 2)

        def normalize_scores(results, score_key):
            if not results:
                return results
            scores = [r[score_key] for r in results]
            max_score = max(scores)
            if max_score > 0:
                for r in results:
                    r[f'{score_key}_normalized'] = r[score_key] / max_score
            return results

        bm25_results = normalize_scores(bm25_results, 'bm25_score')
        tfidf_results = normalize_scores(tfidf_results, 'tfidf_score')
        semantic_results = normalize_scores(semantic_results, 'semantic_score')

        combined_scores = defaultdict(dict)

        for result in bm25_results:
            aid = result['article_id']
            combined_scores[aid]['bm25'] = result.get('bm25_score_normalized', 0)
            combined_scores[aid]['article'] = result

        for result in tfidf_results:
            aid = result['article_id']
            if aid not in combined_scores:
                combined_scores[aid]['article'] = result
            combined_scores[aid]['tfidf'] = result.get('tfidf_score_normalized', 0)

        for result in semantic_results:
            aid = result['article_id']
            if aid not in combined_scores:
                combined_scores[aid]['article'] = result
            combined_scores[aid]['semantic'] = result.get('semantic_score_normalized', 0)

        final_results = []
        for aid, data in combined_scores.items():
            bm25_score = data.get('bm25', 0)
            tfidf_score = data.get('tfidf', 0)
            semantic_score = data.get('semantic', 0)

            hybrid_score = (
                    weights['bm25'] * bm25_score +
                    weights['tfidf'] * tfidf_score +
                    weights['semantic'] * semantic_score
            )

            result = data['article'].copy()
            result['hybrid_score'] = hybrid_score
            final_results.append(result)

        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return final_results[:k]

    def search(self, query, method='hybrid', k=10):
        """main search interface"""
        print(f"\n🔍 searching: '{query}' using {method} method")

        if method == 'bm25':
            results = self.bm25_search(query, k)
        elif method == 'tfidf':
            results = self.tfidf_search(query, k)
        elif method == 'semantic':
            results = self.semantic_search(query, k)
        elif method == 'hybrid':
            results = self.hybrid_search(query, k)
        else:
            print(f"❌ unknown search method: {method}")
            return []

        self.search_logs.append({
            'query': query,
            'method': method,
            'results_count': len(results),
            'top_categories': [r['category'] for r in results[:3]]
        })

        print(f"📊 found {len(results)} results:")
        for i, result in enumerate(results[:5], 1):
            score_key = f"{method}_score" if method != 'hybrid' else 'hybrid_score'
            score = result.get(score_key, 0)
            print(f"   {i}. {result['title'][:60]}...")
            print(f"      category: {result['category']} | source: {result['source']} | score: {score:.4f}")

        return results


class ScalableSearchEngine:
    """
    memory-efficient search engine for 10k+ documents
    demonstrates: distributed processing, chunked indexing, scalable architecture
    """

    def __init__(self, chunk_size=1000, max_workers=8):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.index_cache_path = 'data/indices/'
        self.articles = []

        os.makedirs(self.index_cache_path, exist_ok=True)
        print(f"🔍 scalable search engine initialized")
        print(f"💾 memory usage: {psutil.virtual_memory().percent:.1f}%")

    def load_articles(self, articles_file):
        """load articles from file"""
        print(f"📚 loading articles from {articles_file}...")

        with open(articles_file, 'r', encoding='utf-8') as f:
            self.articles = json.load(f)

        print(f"✅ loaded {len(self.articles)} articles")

    def build_scalable_indices(self):
        """build search indices with memory optimization"""
        if not self.articles:
            print("❌ no articles loaded")
            return

        print(f"🔧 building scalable indices for {len(self.articles)} articles...")

        num_chunks = (len(self.articles) + self.chunk_size - 1) // self.chunk_size
        print(f"📊 processing {num_chunks} chunks of {self.chunk_size} articles each")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(self.articles))
            chunk_articles = self.articles[start_idx:end_idx]

            print(f"   processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_articles)} articles)")

            chunk_indices = self._build_chunk_indices(chunk_articles, chunk_idx)
            self._save_chunk_indices(chunk_indices, chunk_idx)

            del chunk_indices
            if chunk_idx % 5 == 0:
                memory_percent = psutil.virtual_memory().percent
                print(f"   💾 memory usage: {memory_percent:.1f}%")

        print("✅ scalable indexing complete")

    def bm25_search(self, query, k=10):
        """bm25 search using distributed chunks"""
        print(f"🔍 BM25 search for: '{query}'")
        return self.distributed_search(query, top_k=k)

    def tfidf_search(self, query, k=10):
        """tf-idf search using distributed chunks"""
        print(f"🔍 TF-IDF search for: '{query}'")
        return self.distributed_search(query, top_k=k)

    def _build_chunk_indices(self, chunk_articles, chunk_idx):
        """build indices for article chunk"""
        texts = []
        for article in chunk_articles:
            text = f"{article['title']} {article.get('content', '')}"
            texts.append(text.lower())

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        tfidf_matrix = vectorizer.fit_transform(texts)

        return {
            'chunk_idx': chunk_idx,
            'articles': chunk_articles,
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'article_count': len(chunk_articles)
        }

    def _save_chunk_indices(self, chunk_indices, chunk_idx):
        """save chunk indices to disk"""
        chunk_file = f"{self.index_cache_path}/chunk_{chunk_idx}.pkl"

        with open(chunk_file, 'wb') as f:
            pickle.dump(chunk_indices, f)

    def bm25_search(self, query, k=10):
        """bm25 search using distributed chunks"""
        print(f"🔍 BM25 search for: '{query}'")
        return self.distributed_search(query, top_k=k)

    def tfidf_search(self, query, k=10):
        """tf-idf search using distributed chunks"""
        print(f"🔍 TF-IDF search for: '{query}'")
        return self.distributed_search(query, top_k=k)
        """distributed search across all chunks"""
        print(f"🔍 distributed search for: '{query}'")

        chunk_files = [f for f in os.listdir(self.index_cache_path) if f.startswith('chunk_')]

        if not chunk_files:
            print("❌ no index chunks found - run build_scalable_indices() first")
            return []

        with ProcessPoolExecutor(max_workers=min(len(chunk_files), self.max_workers)) as executor:
            search_func = partial(self._search_chunk, query=query, top_k=top_k // 2)
            chunk_results = list(executor.map(search_func, chunk_files))

        all_results = []
        for chunk_result in chunk_results:
            if chunk_result:
                all_results.extend(chunk_result)

        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        print(f"📊 found {len(all_results[:top_k])} results from distributed search")
        return all_results[:top_k]

    def _search_chunk(self, chunk_file, query, top_k):
        """search within a single chunk"""
        try:
            chunk_path = f"{self.index_cache_path}/{chunk_file}"

            with open(chunk_path, 'rb') as f:
                chunk_indices = pickle.load(f)

            vectorizer = chunk_indices['vectorizer']
            tfidf_matrix = chunk_indices['tfidf_matrix']
            articles = chunk_indices['articles']

            query_vector = vectorizer.transform([query.lower()])
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    result = articles[idx].copy()
                    result['score'] = float(similarities[idx])
                    result['chunk_id'] = chunk_indices['chunk_idx']
                    results.append(result)

            return results

        except Exception as e:
            print(f"   ❌ error searching chunk {chunk_file}: {e}")
            return []


# standalone execution for testing
if __name__ == "__main__":
    config = Config()

    # test basic search engine
    print("🧪 testing basic search engine...")
    basic_engine = MultiModalSearchEngine(config.OPENAI_API_KEY)

    # check if we have data
    test_data_path = f"{config.DATA_DIR}/raw/reddit_articles.json"
    if os.path.exists(test_data_path):
        basic_engine.load_articles(test_data_path)

        # test searches
        test_queries = ["artificial intelligence", "climate change", "sports news"]
        for query in test_queries:
            results = basic_engine.search(query, method='hybrid', k=3)
            print()
    else:
        print(f"❌ no test data found at {test_data_path}")
        print("   run data_collector.py first to collect data")

    print("✅ search engine test complete")