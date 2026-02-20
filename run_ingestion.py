import os
from src.ingestion import PolicyIngestionPipeline

if __name__ == "__main__":
    # Define absolute paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
    DB_DIR = os.path.join(BASE_DIR, "data", "chromadb_store")

    # Ensure directories exist
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    # Run the pipeline
    pipeline = PolicyIngestionPipeline(pdf_dir=PDF_DIR, db_dir=DB_DIR)
    pipeline.process_all_pdfs()