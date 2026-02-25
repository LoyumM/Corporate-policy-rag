import os
import hashlib
import sqlite3
import time
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

class PolicyRetriever:
    def __init__(self, db_dir: str = "data/chromadb_store", cache_db_path: str = "data/cache.db"):
        # Initializing Vector Search Components
        print("Loading embedding model for retrieval...")
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # Initializing the Re-Ranker (Cross-Encoder)
        print("Loading Cross-Encoder model for re-ranking...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        print("Connecting to local ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chroma_client.get_collection("corporate_policies")
        
        # Initializing the SQLite Caching Layer
        self.cache_db_path = cache_db_path
        self._init_sqlite_cache()

    def _init_sqlite_cache(self):
        """Creates the SQLite cache database and table if they do not exist."""
        print(f"Connecting to SQLite cache at {self.cache_db_path}...")
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        
        # check_same_thread=False is required so FastAPI can access it concurrently
        with sqlite3.connect(self.cache_db_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT,
                    expiry_time REAL
                )
            """)
            conn.commit()

    def _generate_cache_key(self, query: str) -> str:
        """Generates a deterministic SHA-256 hash of the query for safe caching."""
        normalized_query = query.strip().lower()
        return hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()

    def get_cached_response(self, query: str) -> Optional[str]:
        """Checks the SQLite database for a valid, non-expired cached response."""
        cache_key = self._generate_cache_key(query)
        current_time = time.time()
        
        with sqlite3.connect(self.cache_db_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response, expiry_time FROM response_cache WHERE cache_key = ?", 
                (cache_key,)
            )
            result = cursor.fetchone()
            
            if result:
                response, expiry_time = result
                if current_time < expiry_time:
                    return response
                else:
                    # Silently delete expired cache entries to save disk space
                    cursor.execute("DELETE FROM response_cache WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                    
        return None

    def cache_response(self, query: str, response: str, ttl_seconds: int = 86400):
        """Saves the LLM response to SQLite with a Time-To-Live (24 hours default)."""
        cache_key = self._generate_cache_key(query)
        expiry_time = time.time() + ttl_seconds  # Calculate the exact Unix timestamp of expiration
        
        with sqlite3.connect(self.cache_db_path, check_same_thread=False) as conn:
            cursor = conn.cursor()
            # INSERT OR REPLACE acts as an UPSERT: inserts new, or updates if the key exists
            cursor.execute("""
                INSERT OR REPLACE INTO response_cache (cache_key, response, expiry_time)
                VALUES (?, ?, ?)
            """, (cache_key, response, expiry_time))
            conn.commit()

    # def retrieve_context(self, query: str, top_k: int = 3) -> str:
    #     """Embeds the query and fetches the most relevant document chunks."""
    #     query_embedding = self.embedding_model.encode(query).tolist()
        
    #     results = self.collection.query(
    #         query_embeddings=[query_embedding],
    #         n_results=top_k
    #     )
        
    #     if not results['documents'][0]:
    #         return ""
            
    #     # Combine the chunks into a single string to feed to the LLM
    #     return "\n\n---\n\n".join(results['documents'][0])

    def retrieve_context(self, query: str, initial_k: int = 15, final_k: int = 3) -> str:
        """
        Two-Stage Retrieval Pipeline:
        1. Embeds the query and fetches the top 'initial_k' chunks using the fast Bi-Encoder.
        2. Passes the chunks to a Cross-Encoder to calculate an exact relevance score.
        3. Returns the top 'final_k' re-ranked chunks.
        """
        # Stage 1: Fast Retrieval (Bi-Encoder)
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True # added normalization
            ).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k
        )
        
        retrieved_docs = results['documents'][0]
        
        if not retrieved_docs:
            return ""
            
        # Stage 2: Re-Ranking (Cross-Encoder)
        # The Cross-Encoder requires pairs in the format: [[query, document_1], [query, document_2], ...]
        cross_encoder_inputs = [[query, doc] for doc in retrieved_docs]
        
        # Predict generates a highly accurate relevance score for each pair
        scores = self.cross_encoder.predict(cross_encoder_inputs)
        
        # Combine the scores with their respective documents
        scored_docs = list(zip(scores, retrieved_docs))
        
        # Sort the documents by score in descending order (highest score first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Extract the text of the top 'final_k' documents
        best_docs = [doc for score, doc in scored_docs[:final_k]]
            
        # Combine the strictly filtered chunks into a single string for the LLM prompt
        return "\n\n---\n\n".join(best_docs)