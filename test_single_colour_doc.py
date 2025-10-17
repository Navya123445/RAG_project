"""
test_single_color_doc.py
Test the color-aware RAG pipeline with a single document
"""

from ingestion import ingestion_Documents
from json_transformer import transform_to_structured_json
from annotation_engine import annotate_documents
from splitter import text_splitter
from embedding_vector_doc import embedding_and_vector
from retrieval_and_generation import retrieve_and_generate
import os
import json
from datetime import datetime
import shutil


def test_single_document(pdf_path="40_SPAs/31.RECTORSEAL, LLC (1).pdf"):
    """Test pipeline with single color-coded document"""
    
    start_time = datetime.now()
    
    print("=" * 70)
    print("🎨 SINGLE DOCUMENT COLOR-AWARE TEST")
    print("=" * 70)
    print(f"Testing with: {pdf_path}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create test folder with single doc
    test_folder = "test_single_doc"
    os.makedirs(test_folder, exist_ok=True)
    
    # Copy your PDF to test folder
    if os.path.exists(pdf_path):
        test_pdf_path = os.path.join(test_folder, os.path.basename(pdf_path))
        if not os.path.exists(test_pdf_path):
            shutil.copy(pdf_path, test_pdf_path)
        print(f"✅ PDF copied to {test_folder}/")
    else:
        print(f"❌ PDF not found: {pdf_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Looking for: {os.path.abspath(pdf_path)}")
        return None, None
    
    # Stage 1: Ingestion with color extraction
    print("\n" + "━" * 70)
    print("[1/6] 📄 INGESTION (Color Extraction)")
    print("━" * 70)
    documents = ingestion_Documents(test_folder)
    
    if not documents:
        print("❌ No documents loaded")
        return None, None
    
    # Print color metadata summary
    print("\n🎨 COLOR METADATA EXTRACTED:")
    for i, doc in enumerate(documents[:3]):  # Show first 3 pages
        counts = doc.metadata.get("entity_counts", {})
        print(f"   Page {doc.metadata.get('page', i+1)}: {counts}")
    
    # Stage 2: JSON transformation
    print("\n" + "━" * 70)
    print("[2/6] 🔄 JSON TRANSFORMATION (Color-Prioritized)")
    print("━" * 70)
    structured_documents = transform_to_structured_json(documents)
    
    # Stage 3: Annotation
    print("\n" + "━" * 70)
    print("[3/6] 🏷️  ANNOTATION (Color-Integrated)")
    print("━" * 70)
    annotated_documents = annotate_documents(structured_documents)
    
    # Print annotation sample
    if annotated_documents:
        try:
            sample_annotations = json.loads(annotated_documents[0].metadata.get("annotations", "{}"))
            print("\n📊 SAMPLE ANNOTATIONS (Page 1):")
            print(f"   Companies: {len(sample_annotations.get('legal_entities', {}).get('companies', []))}")
            print(f"   Amounts: {len(sample_annotations.get('financial_information', {}).get('monetary_amounts', []))}")
            print(f"   Confidence: {sample_annotations.get('confidence_scores', {}).get('overall_confidence', 0):.2f}")
        except Exception as e:
            print(f"   ⚠️ Could not parse annotations: {e}")
    
    # Stage 4: Chunking
    print("\n" + "━" * 70)
    print("[4/6] ✂️  CHUNKING (Color-Aware)")
    print("━" * 70)
    doc_chunks = text_splitter(annotated_documents)
    
    # Print chunk color stats
    color_chunks = sum(1 for c in doc_chunks if c.metadata.get("color_entity_count", 0) > 0)
    amount_chunks = sum(1 for c in doc_chunks if c.metadata.get("has_color_amounts", False))
    party_chunks = sum(1 for c in doc_chunks if c.metadata.get("has_color_parties", False))
    
    print(f"\n🎨 CHUNK COLOR STATISTICS:")
    print(f"   Total chunks: {len(doc_chunks)}")
    print(f"   With color entities: {color_chunks}")
    print(f"   With color-coded amounts: {amount_chunks}")
    print(f"   With color-coded parties: {party_chunks}")
    
    # Stage 5: Embeddings
    print("\n" + "━" * 70)
    print("[5/6] 🔢 EMBEDDINGS (Color-Tracked)")
    print("━" * 70)
    vector_store = embedding_and_vector(doc_chunks)
    
    # Stage 6: RAG setup
    print("\n" + "━" * 70)
    print("[6/6] 🚀 RAG PIPELINE SETUP")
    print("━" * 70)
    rag_pipeline = retrieve_and_generate(vector_store)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"📄 Pages: {len(documents)}")
    print(f"🧩 Chunks: {len(doc_chunks)}")
    if len(doc_chunks) > 0:
        print(f"🎨 Color chunks: {color_chunks}/{len(doc_chunks)} ({color_chunks/len(doc_chunks)*100:.1f}%)")
    
    # Test queries - RECTORSEAL SPECIFIC
    print("\n" + "=" * 70)
    print("🔍 RUNNING RECTORSEAL-SPECIFIC TEST QUERIES")
    print("=" * 70)
    
    test_queries = [
        "What is the Stock Consideration component and Parent Stock listing requirements in the RectorSeal CSW Industrials transaction?",
        "What are the Reserve Account provisions and Seller Representative expense coverage in the RectorSeal agreement?",
        "What RWI Policy coverage and certificate requirements apply at Closing in the RectorSeal deal?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"[Query {i}/3]")
        print(f"{'='*70}")
        print(f"Q: {query}\n")
        
        try:
            print("⏳ Querying RAG pipeline...")
            answer = rag_pipeline.invoke(query)
            
            print(f"✅ ANSWER ({len(answer)} characters):")
            print(f"{'-'*70}")
            print(answer)
            print(f"{'-'*70}\n")
            
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
    
    print("\n" + "=" * 70)
    print("🎨 COLOR-AWARE PIPELINE TEST COMPLETE")
    print("=" * 70)
    
    return vector_store, rag_pipeline


def inspect_color_metadata(pdf_path="40_SPAs/31.RECTORSEAL, LLC (1).pdf"):
    """Quick inspection of color metadata only"""
    
    print("\n" + "=" * 70)
    print("🎨 COLOR METADATA INSPECTION")
    print("=" * 70)
    
    test_folder = "test_single_doc"
    os.makedirs(test_folder, exist_ok=True)
    
    test_pdf_path = os.path.join(test_folder, os.path.basename(pdf_path))
    if os.path.exists(pdf_path) and not os.path.exists(test_pdf_path):
        shutil.copy(pdf_path, test_pdf_path)
    
    documents = ingestion_Documents(test_folder)
    
    if not documents:
        print("❌ No documents loaded")
        return
    
    print(f"\n📊 Document: {os.path.basename(pdf_path)}")
    print(f"📄 Total pages: {len(documents)}\n")
    
    # Aggregate color statistics
    all_categories = {}
    for doc in documents:
        counts = doc.metadata.get("entity_counts", {})
        for category, count in counts.items():
            all_categories[category] = all_categories.get(category, 0) + count
    
    print("🎨 TOTAL COLOR ENTITIES BY CATEGORY:")
    if all_categories:
        for category, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {category:15s}: {count:3d}")
    else:
        print("   ⚠️ No color entities found")
    
    # Show sample entities from first page
    if documents:
        first_page = documents[0]
        categories = first_page.metadata.get("color_categories", {})
        
        if isinstance(categories, str):
            try:
                categories = json.loads(categories)
            except:
                categories = {}
        
        print(f"\n🔍 SAMPLE ENTITIES (Page 1):")
        if categories:
            for category, entities in categories.items():
                if entities:
                    sample = entities[:3] if len(entities) > 3 else entities
                    print(f"   {category}: {sample}")
        else:
            print("   ⚠️ No color categories found on page 1")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    # Get PDF path from command line or use default
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "40_SPAs/31.RECTORSEAL, LLC (1).pdf"
    
    # Quick color inspection first
    inspect_color_metadata(pdf_path)
    
    # Full pipeline test
    print("\n\n")
    vector_store, rag_pipeline = test_single_document(pdf_path)
    
    if vector_store and rag_pipeline:
        print("\n" + "=" * 70)
        print("💾 INTERACTIVE MODE READY")
        print("=" * 70)
        print("\n🎯 Try custom queries:")
        print("   python -i test_single_color_doc.py")
        print("   >>> answer = rag_pipeline.invoke('your question')")
        print("   >>> print(answer)")
        print("\n📝 Example queries for RectorSeal:")
        print("   • 'What is the purchase price structure?'")
        print("   • 'What are the escrow arrangements?'")
        print("   • 'What are the indemnification caps?'")
        print("=" * 70)
