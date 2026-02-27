import requests
from typing import List, Dict, Any
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class GlobalTrendsCrawler:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://google.serper.dev/search"
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

    def discover_from_seeds(self, seeds: List[str], geo: str = 'ke') -> List[Dict[str, Any]]:
        """
        Discovers related keywords and questions from a set of seed queries.
        """
        all_discovered = []
        for seed in seeds:
            try:
                logger.info(f"Discovering from seed: {seed} ({geo})")
                payload = {
                    "q": seed,
                    "gl": geo,
                    "hl": "en",
                    "autocorrect": True
                }
                
                response = requests.post(self.url, headers=self.headers, json=payload, timeout=15)
                if response.status_code != 200:
                    logger.error(f"Serper request failed: {response.status_code}")
                    continue
                
                data = response.json()
                
                # 1. Related Searches
                if 'relatedSearches' in data:
                    for item in data['relatedSearches']:
                        all_discovered.append({
                            "keyword": item.get('query'),
                            "origin_seed": seed,
                            "geo": geo,
                            "type": "related_search",
                            "discovered_at": datetime.now().isoformat()
                        })
                
                # 2. People Also Ask
                if 'peopleAlsoAsk' in data:
                    for item in data['peopleAlsoAsk']:
                        all_discovered.append({
                            "keyword": item.get('question'),
                            "origin_seed": seed,
                            "geo": geo,
                            "type": "question",
                            "discovered_at": datetime.now().isoformat()
                        })
                
                # 3. Organic results extraction (titles/snippets for entities)
                if 'organic' in data:
                    for item in data['organic'][:3]: # Only top 3
                        # We could extract more here, but let's stick to explicit related queries for now
                        pass

            except Exception as e:
                logger.error(f"Error discovering from seed {seed}: {e}")
        
        return all_discovered

def discover_global_trends(api_key: str, geo: str = 'ke', niches: List[str] = None) -> List[Dict[str, Any]]:
    # System default seeds if no niches provided
    if not niches:
        niches = [
            "trending topics kenya",
            "business news kenya",
            "tech trends 2026",
            "e-commerce trends worldwide",
            "cryptocurrency news global"
        ]
    
    crawler = GlobalTrendsCrawler(api_key)
    return crawler.discover_from_seeds(niches, geo)
