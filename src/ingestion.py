import os
from pathlib import Path
from typing import List

import chromadb
from docling.document_converter import DocumentConverter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer


class PolicyIngestionPipeline:
    def __init__(self, pdf_dir: str, db_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.db_dir = Path(db_dir)
        
        self.converter = DocumentConverter()
        
        # Define which markdown headers we want to split on
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        
        # Fallback character splitter: In case a single section under a header is massively long
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, 
            chunk_overlap=100
        )

        # 2. Initializing the Embedding Model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # 3. Initialize Persistent ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name="corporate_policies",
            metadata={"hnsw:space": "cosine"} # Optimize for cosine similarity
        )

    def process_all_pdfs(self):
        """Finds all PDFs in the target directory and processes them."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {self.pdf_dir}")
            return

        for pdf_path in pdf_files:
            print(f"\nProcessing: {pdf_path.name}")
            self._process_single_pdf(pdf_path)
            
        print("\nAll documents ingested successfully!")

    def _process_single_pdf(self, pdf_path: Path):
        # Converting PDF to Markdown using Docling
        print("  -> Converting to Markdown...")
        result = self.converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()

        # Semantic Split by Headers
        print("  -> Chunking by semantic headers...")
        header_splits = self.markdown_splitter.split_text(markdown_text)

        # Apply fallback splitting to oversized sections and prepare data
        final_chunks = []
        final_metadatas = []
        final_ids = []

        for i, split in enumerate(header_splits):
            # If the section is too large, break it down further
            sub_chunks = self.fallback_splitter.split_text(split.page_content)
            
            for j, sub_chunk in enumerate(sub_chunks):
                final_chunks.append(sub_chunk)
                
                # Combine original document metadata with the extracted header metadata
                meta = split.metadata.copy()
                meta["source_file"] = pdf_path.name
                final_metadatas.append(meta)
                
                # Create a unique ID for the vector database
                final_ids.append(f"{pdf_path.stem}_chunk_{i}_{j}")

        # Generate Embeddings
        print(f"  -> Generating embeddings for {len(final_chunks)} chunks...")
        embeddings = self.embedding_model.encode(final_chunks).tolist()

        # Store in ChromaDB
        print("  -> Storing in ChromaDB...")
        self.collection.upsert(
            documents=final_chunks,
            embeddings=embeddings,
            metadatas=final_metadatas,
            ids=final_ids
        )