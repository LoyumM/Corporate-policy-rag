import time
from collections import deque

class MetricsTracker:
    def __init__(self, window_size=500):
        self.window_size = window_size

        self.cache_hits = 0
        self.cache_misses = 0

        self.retrieval_latencies = deque(maxlen=window_size)
        self.llm_latencies = deque(maxlen=window_size)

        self.similarity_scores = deque(maxlen=window_size)

    # ---- Cache Metrics ----

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def cache_hit_ratio(self):
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    # ---- Latency Metrics ----

    def record_retrieval_latency(self, ms):
        self.retrieval_latencies.append(ms)

    def record_llm_latency(self, ms):
        self.llm_latencies.append(ms)

    # ---- Similarity Metrics ----

    def record_similarity_scores(self, scores):
        for s in scores:
            self.similarity_scores.append(s)

    def similarity_distribution(self):
        if not self.similarity_scores:
            return None

        scores = list(self.similarity_scores)
        return {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores)
        }