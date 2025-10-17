from ingestion import ingestion_Documents
from json_transformer import transform_to_structured_json
from annotation_engine import annotate_documents  # ← ADD THIS IMPORT
from splitter import text_splitter
from embedding_vector_doc import embedding_and_vector
from retrieval_and_generation import retrieve_and_generate
import os

def main():
    """Main orchestration script for SPA RAG system setup"""
    print("="*60)
    print("SPA RAG SYSTEM INITIALIZATION")
    print("="*60)
    
    # Stage 1: Document Ingestion
    print("\n[1/6] Loading SPA documents...")  # ← CHANGE FROM [1/5]
    documents = ingestion_Documents("40_SPAs")
    
    # Stage 2: JSON Transformation
    print("\n[2/6] Transforming to structured JSON...")  # ← CHANGE FROM [2/5]
    structured_documents = transform_to_structured_json(documents)
    
    # Stage 3: Legal Annotation (NEW STAGE)
    print("\n[3/6] Annotating legal entities and financial terms...")  # ← ADD THIS
    annotated_documents = annotate_documents(structured_documents)  # ← ADD THIS
    
    # Stage 4: Text Chunking
    print("\n[4/6] Chunking documents...")  # ← CHANGE FROM [3/5]
    doc_chunks = text_splitter(annotated_documents)  # ← CHANGE INPUT FROM structured_documents
    
    # Stage 5: Vector Embeddings & Storage
    print("\n[5/6] Creating embeddings and vector store...")  # ← CHANGE FROM [4/5]
    vector_store = embedding_and_vector(doc_chunks)
    
    # Stage 6: RAG Pipeline Setup
    print("\n[6/6] Setting up retrieval and generation pipeline...")  # ← CHANGE FROM [5/5]
    rag_pipeline = retrieve_and_generate(vector_store)
    
    print("\n" + "="*60)
    print("✅ ENHANCED SPA RAG SYSTEM READY WITH ANNOTATIONS")  # ← UPDATE MESSAGE
    print("Use 'python batch_query_test.py' to start querying")
    print("="*60)
    
    return vector_store, rag_pipeline

if __name__ == "__main__":
    main()
