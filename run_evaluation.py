import os
import json
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    Faithfulness,
    AnswerCorrectness
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv 
from src.retrieval import PolicyRetriever
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Missing OPENAI_API_KEY. Please check your .env file.")

# Configure Ragas
judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3.5:latest"

print("Initializing local retrieval system...")
retriever = PolicyRetriever()

def get_ollama_answer(query: str, context: str) -> str:
    """Synchronous helper function to get an answer from the local Phi-3.5 model."""
    combined_prompt = f"""You are an expert HR and Corporate Policy Assistant. 
    Answer the user's question directly and concisely using ONLY the provided context below.
    If the answer is not contained in the context, explicitly state "I cannot find this information in the provided policies."
    
    CONTEXT:
    {context}
    
    User Question: {query}"""
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": combined_prompt}],
        "options": {"num_ctx": 4096},
        "stream": False # We turn off streaming for evaluation to get the whole answer at once
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]

def main():
    # Loading the Golden Dataset
    print("Loading golden_dataset.json...")
    with open("data\golden_dataset.json", "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    questions = []
    contexts_list = []
    generated_answers = []
    ground_truths = []

    # Generating Traces (The local system takes the exam)
    print(f"Running evaluation on {len(golden_data)} questions. This will take a few minutes...")
    for idx, item in enumerate(golden_data):
    # for idx, item in enumerate(golden_data[:3]): #trial
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"Processing ({idx+1}/{len(golden_data)}): {question}")
        
        # Retrieve context
        raw_context = retriever.retrieve_context(question)

        if isinstance(raw_context, tuple):
            raw_context = raw_context[0]
        
        # Ragas requires a list of strings for contexts, so we split our combined string back into a list
        if raw_context:
            contexts = raw_context.split("\n\n---\n\n")
        else:
            contexts = [""]
            
        # Generate answer
        answer = get_ollama_answer(question, raw_context)
        
        # Record the trace
        questions.append(question)
        contexts_list.append(contexts)
        generated_answers.append(answer)
        ground_truths.append(ground_truth)

    # Format data for Ragas
    data_dict = {
        "question": questions,
        "contexts": contexts_list,
        "answer": generated_answers,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data_dict)

    # Run the LLM-as-a-Judge Evaluation
    print("\nTraces collected! Sending data to gpt-4o-mini for grading...")

    # Initialize the Classes and inject the OpenAI Judge directly into them
    metrics_list = [
        ContextPrecision(llm=judge_llm),
        Faithfulness(llm=judge_llm),
        AnswerCorrectness(llm=judge_llm, embeddings=judge_embeddings)
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics_list,
    )
    

    # Save and Display Results
    print("\n=== EVALUATION COMPLETE ===")
    print(result)
    
    # Export to a CSV so you can inspect individual failures
    df = result.to_pandas()
    df.to_csv("data\evaluation_results.csv", index=False)
    print("\nDetailed results saved to 'evaluation_results.csv'.")

if __name__ == "__main__":
    main()