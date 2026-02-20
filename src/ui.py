import gradio as gr
import requests
import json

API_URL = "http://localhost:8000/api/ask"

def chat_with_policy_bot(message, history):
    """Sends the user message to the FastAPI backend and yields the streaming response."""
    
    payload = {"query": message}
    
    try:
        # Using standard requests with stream=True to catch the Server-Sent Events
        with requests.post(API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            
            partial_message = ""
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    partial_message += chunk
                    # Yielding the string updates the Gradio UI in real-time
                    yield partial_message
                    
    except requests.exceptions.ConnectionError:
        yield "Error: Could not connect to the API. Is FastAPI running on port 8000?"
    except Exception as e:
        yield f"An unexpected error occurred: {str(e)}"

demo = gr.ChatInterface(
    fn=chat_with_policy_bot,
    title="Corporate Policy RAG Assistant",
    description="Ask questions about company travel, reimbursement, and HR policies.",
    examples=[
        "What are some non-allowable travel expenses?",
        "What is the policy on spousal travel?",
        "What is the policy on booking flights?"
    ]
)

if __name__ == "__main__":
    # Launch on port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)