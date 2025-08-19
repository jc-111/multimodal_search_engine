# data_collector.py - 數據收集模塊
# scalable data collection for news articles and reddit posts

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import count
from threading import Lock
from typing import List, Dict, Optional

import pandas as pd
import requests

from config import Config


class NewsDataCollector:
    """
    Original news data collector using NewsAPI with basic quality control,
    timeout/retries, and standardized output.
    """

    def __init__(self, api_key: str, request_timeout: int = 10, max_retries: int = 3):
        self.api_key = api_key
        self.collected_articles: List[Dict] = []
        self.failed_articles: List[Dict] = []
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._user_agent = "MultimodalSearchEngine/1.0 (+https://github.com/jc-111)"

    def collect_articles_by_category(self, category: str, target_count: int = 20) -> List[Dict]:
        """Collect articles from a specific category with quality control."""
        print(f"\n📰 collecting {category} articles...")

        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": self.api_key,
            "category": category,
            "language": "en",
            "pageSize": 30,  # fetch extra for filtering
            "sortBy": "publishedAt",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(
                    url, params=params, headers={"User-Agent": self._user_agent}, timeout=self.request_timeout
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                data = resp.json() or {}
                articles = data.get("articles", [])
                print(f"📊 raw articles received: {len(articles)}")

                quality_articles: List[Dict] = []
                for article in articles:
                    if self._is_quality_article(article):
                        processed = self._process_article(article, category)
                        quality_articles.append(processed)
                        if len(quality_articles) >= target_count:
                            break
                    else:
                        self.failed_articles.append(
                            {
                                "title": article.get("title", "no title"),
                                "reason": self._get_rejection_reason(article),
                            }
                        )

                print(f"✅ quality articles collected: {len(quality_articles)}")
                return quality_articles

            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    print(f"❌ error collecting {category}: {e}")

        return []

    def _is_quality_article(self, article: Dict) -> bool:
        """Quality control filter."""
        if not article.get("urlToImage"):
            return False
        content = (article.get("content") or "").strip()
        if len(content) < 200:
            return False
        title = (article.get("title") or "").strip()
        if len(title) < 10 or "[Removed]" in title:
            return False
        junk_sources = {"[Removed]", "Google News"}
        source_name = (article.get("source") or {}).get("name", "")
        if source_name in junk_sources:
            return False
        return True

    def _get_rejection_reason(self, article: Dict) -> str:
        """Track rejection reasons."""
        if not article.get("urlToImage"):
            return "no_image"
        elif len((article.get("content") or "").strip()) < 200:
            return "insufficient_content"
        elif "[Removed]" in (article.get("title") or ""):
            return "removed_content"
        else:
            return "other"

    def _process_article(self, article: Dict, category: str) -> Dict:
        """Process raw article into standardized format."""
        return {
            "id": len(self.collected_articles) + 1,  # local incremental id
            "title": article.get("title", "") or "",
            "content": article.get("content", "") or "",
            "description": article.get("description", "") or "",
            "url": article.get("url", "") or "",
            "image_url": article.get("urlToImage", "") or "",
            "published_at": article.get("publishedAt", "") or "",
            "category": category,
            "source": (article.get("source") or {}).get("name", "") or "",
            "collected_at": datetime.now().isoformat(),
        }

    def collect_all_categories(self) -> List[Dict]:
        """Collect articles from all categories."""
        categories = ["technology", "business", "science", "health", "sports"]
        print("🚀 starting news data collection...")

        for category in categories:
            category_articles = self.collect_articles_by_category(category, 20)
            self.collected_articles.extend(category_articles)
            time.sleep(1)  # respectful API usage

        print(f"🎉 total articles collected: {len(self.collected_articles)}")
        return self.collected_articles


class ScalableDataCollector:
    """
    Scalable data collector for 10k+ articles using Reddit (public JSON) with
    distributed fetching, retries, and thread-safe ids.
    """

    def __init__(self, max_workers: int = 8, request_timeout: int = 10, max_retries: int = 3):
        self.max_workers = max_workers
        self.batch_size = 100
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._user_agent = "MultimodalSearchEngine/1.0 (+https://github.com/jc-111)"
        # Thread-safe id generator
        self._id_lock = Lock()
        self._id_counter = count(start=0)
        print(f"🚀 initialized scalable collector with {self.max_workers} workers")

    def _next_article_id(self) -> int:
        with self._id_lock:
            return next(self._id_counter)

    def collect_reddit_data(self, target_articles: int = 10000) -> List[Dict]:
        """
        Collect large-scale Reddit data with parallel processing.
        """
        print(f"📊 collecting {target_articles} articles from reddit...")

        # Popular subreddits with good content diversity
        subreddits = [
            "technology",
            "MachineLearning",
            "artificial",
            "programming",
            "science",
            "Physics",
            "biology",
            "chemistry",
            "space",
            "worldnews",
            "news",
            "politics",
            "economics",
            "business",
            "health",
            "medicine",
            "fitness",
            "nutrition",
            "psychology",
            "sports",
            "nfl",
            "soccer",
            "basketball",
            "baseball",
        ]

        articles_per_subreddit = max(1, target_articles // max(1, len(subreddits)))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._collect_subreddit_posts, sr, articles_per_subreddit) for sr in subreddits
            ]

            all_articles: List[Dict] = []
            for fut in futures:
                try:
                    subreddit_articles = fut.result(timeout=300)
                    all_articles.extend(subreddit_articles)
                    print(f"   ✅ collected {len(subreddit_articles)} articles")
                except Exception as e:
                    print(f"   ❌ subreddit collection failed: {e}")

        print(f"🎉 total reddit articles collected: {len(all_articles)}")
        return all_articles

    def _collect_subreddit_posts(self, subreddit: str, target_count: int) -> List[Dict]:
        """Collect posts from a single subreddit with basic retry logic."""
        articles: List[Dict] = []
        base_url = f"https://www.reddit.com/r/{subreddit}/hot.json"
        params = {"limit": min(100, target_count), "raw_json": 1}
        headers = {"User-Agent": self._user_agent}

        tries, backoff = self.max_retries, 1.0
        for attempt in range(tries):
            try:
                time.sleep(1)  # polite rate limit
                resp = requests.get(base_url, params=params, headers=headers, timeout=self.request_timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                data = resp.json() or {}
                posts = data.get("data", {}).get("children", [])

                for post_data in posts[:target_count]:
                    post = post_data.get("data", {}) or {}
                    if self._is_quality_post(post):
                        articles.append(self._process_reddit_post(post, subreddit))
                break
            except Exception as e:
                if attempt == tries - 1:
                    print(f"   ⚠️  error collecting {subreddit}: {e}")
                else:
                    time.sleep(backoff)
                    backoff *= 2

        return articles

    def _is_quality_post(self, post: Dict) -> bool:
        """
        Quality filter:
        - title length >= 10
        - score > 10
        - either: link post (not self) OR self post with selftext >= 100
        - must have a link (permalink or url)
        """
        title_ok = len((post.get("title") or "").strip()) >= 10
        score_ok = (post.get("score") or 0) > 10
        is_self = bool(post.get("is_self", True))
        selftext_len = len((post.get("selftext") or "").strip())
        content_ok = (not is_self) or (selftext_len >= 100)
        has_link = bool(post.get("permalink") or post.get("url"))

        return bool(title_ok and score_ok and content_ok and has_link)

    def _process_reddit_post(self, post: Dict, subreddit: str) -> Dict:
        """Standardize reddit post format."""
        permalink = post.get("permalink")
        url = f"https://reddit.com{permalink}" if permalink else (post.get("url") or "")
        return {
            "id": self._next_article_id(),  # unique, thread-safe
            "title": post.get("title", "") or "",
            "content": post.get("selftext", "") or "",
            "url": url,
            "category": self._map_subreddit_to_category(subreddit),
            "source": f"r/{subreddit}",
            "score": int(post.get("score", 0) or 0),
            "num_comments": int(post.get("num_comments", 0) or 0),
            "created_utc": float(post.get("created_utc", 0) or 0.0),
            "upvote_ratio": float(post.get("upvote_ratio", 0.5) or 0.5),
            "collected_at": datetime.now().isoformat(),
        }

    def _map_subreddit_to_category(self, subreddit: str) -> str:
        """Map subreddit to standard categories."""
        key = (subreddit or "").lower()
        category_map = {
            "technology": "technology",
            "machinelearning": "technology",
            "artificial": "technology",
            "programming": "technology",
            "science": "science",
            "physics": "science",
            "biology": "science",
            "chemistry": "science",
            "space": "science",
            "worldnews": "news",
            "news": "news",
            "politics": "news",
            "economics": "business",
            "business": "business",
            "health": "health",
            "medicine": "health",
            "fitness": "health",
            "nutrition": "health",
            "psychology": "health",
            "sports": "sports",
            "nfl": "sports",
            "soccer": "sports",
            "basketball": "sports",
            "baseball": "sports",
        }
        return category_map.get(key, "general")

    def save_articles(self, articles: List[Dict], filepath: str) -> None:
        """Save collected articles to json file."""
        print(f"💾 saving {len(articles)} articles to {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print("✅ articles saved successfully")


# Standalone execution for quick testing
if __name__ == "__main__":
    config = Config()

    # Test scalable collector (Reddit)
    collector = ScalableDataCollector(max_workers=4)
    articles = collector.collect_reddit_data(target_articles=100)
    collector.save_articles(articles, f"{config.DATA_DIR}/raw/reddit_articles.json")

    # Test NewsAPI collector (requires NEWS_API_KEY in env/.env via Config)
    if config.NEWS_API_KEY:
        news = NewsDataCollector(api_key=config.NEWS_API_KEY)
        news_articles = news.collect_all_categories()
        news_path = f"{config.DATA_DIR}/raw/newsapi_articles.json"
        news.save_articles = collector.save_articles  # reuse saver
        news.save_articles(news_articles, news_path)

    print("✅ data collection test complete")
