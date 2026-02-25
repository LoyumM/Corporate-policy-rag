import os
import json
import httpx
import time
from src.logging_config import setup_logger
from src.metrics import MetricsTracker
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.retrieval import PolicyRetriever

logger = setup_logger()
metrics = MetricsTracker()
logger.info("logger_initialized", extra={"extra_data": {"event": "startup"}}) # temp

app = FastAPI(title="Corporate Policy RAG API")
retriever = PolicyRetriever()

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5:latest") 

class AskRequest(BaseModel):
    query: str

async def stream_ollama_response(query: str, context: str):
    """Async generator that streams tokens from Ollama and handles caching."""
    
    system_prompt = f"""You are an expert HR and Corporate Policy Assistant. 
    Answer the user's question directly and concisely using ONLY the provided context below.
    If the answer is not contained in the context, explicitly state "I cannot find this information in the provided policies."
    
    CONTEXT:
    {context}
    """
    
    payload = {
        "model": "OLLAMA_MODEL", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "stream": True
    }
    
    full_response = ""
    
# Use httpx for asynchronous HTTP requests
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload, timeout=90.0) as response:
                
                # Check for 400 Bad Request (or any error) BEFORE streaming
                if response.status_code != 200:
                    await response.aread() # Safely read the error while stream is open
                    error_msg = response.text
                    yield f"\n[Ollama Error {response.status_code}]: {error_msg}"
                    return # Exit the generator immediately
                
                # If 200 OK, yield tokens as they arrive over the network
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            full_response += token
                            yield token
                            
            # Once the stream is completely finished, cache the final assembled response
            retriever.cache_response(query, full_response)
            
        except httpx.RequestError as e:
            yield f"\n[Error communicating with LLM: {str(e)}]"
        except Exception as e:
            yield f"\n[Unexpected Error]: {str(e)}"

@app.post("/api/ask")
async def ask_question(request: AskRequest):
    query = request.query
    
    # Cache Check: Instant return if previously answered
    cached_answer = retriever.get_cached_response(query)
    if cached_answer:
        # To keep the API interface consistent, we yield the cached response as a single stream chunk
        async def stream_cache():
            yield "[Served from Cache]\n\n" + cached_answer
        return StreamingResponse(stream_cache(), media_type="text/event-stream")
    
    # Retrieval: Get semantic context from ChromaDB
    # context = retriever.retrieve_context(query)
    context, retrieval_metrics = retriever.retrieve_context(query)
    if not context:
        async def no_context():
            yield "I do not have any relevant policy documents to answer that question."
        return StreamingResponse(no_context(), media_type="text/event-stream")
    
    # logging cached answers
    if cached_answer:
        metrics.record_cache_hit()
        logger.info(
            "cache_hit",
            extra={
                "extra_data": {
                    "event": "cache_hit",
                    "hit_ratio": metrics.cache_hit_ratio()
                }
            }
        )
    else:
        metrics.record_cache_miss()

    # logging retrieval latency and similarity scores
    metrics.record_retrieval_latency(retrieval_metrics["total_latency_ms"])
    metrics.record_similarity_scores(retrieval_metrics["similarity_scores"])

    logger.info(
        "retrieval_complete",
        extra={
            "extra_data": {
                "event": "retrieval_complete",
                "bi_latency_ms": retrieval_metrics["bi_latency_ms"],
                "rerank_latency_ms": retrieval_metrics["rerank_latency_ms"],
                "total_latency_ms": retrieval_metrics["total_latency_ms"],
                "similarity_distribution": metrics.similarity_distribution()
            }
        }
    )
        
    # Generation: Stream the LLM reasoning
    return StreamingResponse(stream_ollama_response(query, context), media_type="text/event-stream")

