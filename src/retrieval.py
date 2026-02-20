import os
import json
import hashlib
from typing import List, Dict, Optional
import chromadb
import redis
from sentence_transformers import SentenceTransformer

class PolicyRetriever:
    def __init__(self, db_dir: str = "data/chromadb_store"):
        # Initializing Vector Search Components
        print("Loading embedding model for retrieval...")
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        print("Connecting to local ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chroma_client.get_collection("corporate_policies")
        
        # Initializing the Caching Layer (Redis with in-memory fallback)
        self.redis_client = None
        self.memory_cache = {} # Fallback if Redis isn't running locally
        
        try:
            # Connecting to Redis
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=6379,
                decode_responses=True,
                socket_timeout=2
            )
            self.redis_client.ping() # Test connection
            print("Redis cache connected successfully.")
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, redis.exceptions.ResponseError) as e:
            print(f"Warning: Redis connection failed ({type(e).__name__}). Falling back to in-memory dictionary cache.")
            self.redis_client = None

    def _generate_cache_key(self, query: str) -> str:
        """Generates a deterministic SHA-256 hash of the query for safe caching."""
        normalized_query = query.strip().lower()
        return hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()

    def get_cached_response(self, query: str) -> Optional[str]:
        """Checks if the exact query has been asked and answered previously."""
        cache_key = self._generate_cache_key(query)
        
        if self.redis_client:
            return self.redis_client.get(cache_key)
        else:
            return self.memory_cache.get(cache_key)

    def cache_response(self, query: str, response: str, ttl_seconds: int = 86400):
        """Saves the LLM response to the cache with a Time-To-Live (24 hours default)."""
        cache_key = self._generate_cache_key(query)
        
        if self.redis_client:
            self.redis_client.setex(name=cache_key, time=ttl_seconds, value=response)
        else:
            self.memory_cache[cache_key] = response

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Embeds the query and fetches the most relevant document chunks."""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if not results['documents'][0]:
            return ""
            
        # Combine the chunks into a single string to feed to the LLM
        return "\n\n---\n\n".join(results['documents'][0])