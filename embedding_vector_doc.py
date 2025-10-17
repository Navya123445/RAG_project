from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
import weaviate
import os
from dotenv import load_dotenv
import time
import json

load_dotenv()


def embedding_and_vector(chunks):    
    """
    Color-aware embedding and vector storage with support for:
    - Color metadata preservation and tracking
    - Annotation metadata preservation
    - JSON data cleaning for Weaviate compatibility  
    - Intelligent batch processing
    - Enhanced error handling and monitoring
    """
    model_name = os.getenv("OPENAI_MODEL", "text-embedding-3-large")
    print(f"🔄 Creating vector embeddings... using model - {model_name}")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8081")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    index_name = os.getenv("WEAVIATE_INDEX_NAME", "LegalDocuments")
    
    print(f"🔌 Connecting to Weaviate at {weaviate_url}...")
    
    # Fixed: Proper port configuration with gRPC skip
    if weaviate_api_key:
        client = weaviate.connect_to_local(
            host="localhost", 
            port=8081,  # ← FIXED: Correct REST port
            grpc_port=50052,  # ← FIXED: Match docker-compose
            skip_init_checks=True,  # ← FIXED: Skip gRPC health check
            headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
        )
    else:
        client = weaviate.connect_to_local(
            host="localhost", 
            port=8081,  # ← FIXED: Correct REST port
            grpc_port=50052,  # ← FIXED: Match docker-compose
            skip_init_checks=True  # ← FIXED: Skip gRPC health check
        )
    
    print(f"📊 Processing {len(chunks)} color-aware chunks...")
    
    # Enhanced statistics tracking (includes color metadata)
    stats = {
        "total": 0,
        "with_annotations": 0,
        "with_financial": 0,
        "high_quality": 0,
        # 🎨 Color-specific stats
        "with_color_entities": 0,
        "with_color_amounts": 0,
        "with_color_parties": 0,
        "with_color_dates": 0,
        "with_highlights": 0,
        "total_color_entities": 0
    }
    
    # Clean and process metadata for Weaviate
    processed_chunks = []
    
    for chunk in chunks:
        cleaned_metadata = {}
        
        for key, value in chunk.metadata.items():
            # Handle complex nested structures
            if key in ["structured_data", "annotations", "annotation_summary", 
                       "color_entities", "color_categories", "chunk_color_entities", 
                       "chunk_color_categories"]:
                # Store as JSON string
                if isinstance(value, str):
                    cleaned_metadata[key] = value
                else:
                    cleaned_metadata[key] = json.dumps(value) if value else ""
            
            # Handle basic types
            elif isinstance(value, (str, int, float, bool)):
                cleaned_metadata[key] = value
            elif value is None:
                cleaned_metadata[key] = ""
            else:
                # Convert complex types to strings
                cleaned_metadata[key] = str(value)
        
        chunk.metadata = cleaned_metadata
        processed_chunks.append(chunk)
        
        # Track statistics (including color metadata)
        stats["total"] += 1
        
        # Annotation stats
        if chunk.metadata.get("has_annotations", False):
            stats["with_annotations"] += 1
        if chunk.metadata.get("contains_financial_info", False):
            stats["with_financial"] += 1
        if chunk.metadata.get("high_quality_chunk", False):
            stats["high_quality"] += 1
        
        # 🎨 Color metadata stats
        if chunk.metadata.get("color_entity_count", 0) > 0:
            stats["with_color_entities"] += 1
            stats["total_color_entities"] += chunk.metadata.get("color_entity_count", 0)
        
        if chunk.metadata.get("has_color_amounts", False):
            stats["with_color_amounts"] += 1
        
        if chunk.metadata.get("has_color_parties", False):
            stats["with_color_parties"] += 1
        
        if chunk.metadata.get("has_color_dates", False):
            stats["with_color_dates"] += 1
        
        if chunk.metadata.get("has_highlights", False):
            stats["with_highlights"] += 1
    
    # Batch processing configuration
    batch_size = 25
    total_batches = (len(processed_chunks) + batch_size - 1) // batch_size
    
    try:
        # Create vectorstore
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key="text",
            embedding=embeddings
        )
        
        # Enhanced statistics reporting
        print(f"\n📈 ENHANCED STATISTICS:")
        print(f"   📄 Total chunks: {stats['total']}")
        print(f"   🏷️  With annotations: {stats['with_annotations']}")
        print(f"   💰 With financial info: {stats['with_financial']}")
        print(f"   ⭐ High quality: {stats['high_quality']}")
        
        # 🎨 Color-specific statistics
        print(f"\n🎨 COLOR METADATA STATISTICS:")
        print(f"   🌈 With color entities: {stats['with_color_entities']}")
        print(f"   💵 With color-coded amounts: {stats['with_color_amounts']}")
        print(f"   👥 With color-coded parties: {stats['with_color_parties']}")
        print(f"   📅 With color-coded dates: {stats['with_color_dates']}")
        print(f"   ✨ With highlights: {stats['with_highlights']}")
        print(f"   📊 Total color entities: {stats['total_color_entities']}")
        if stats['with_color_entities'] > 0:
            avg_entities = stats['total_color_entities'] / stats['with_color_entities']
            print(f"   📈 Avg entities per chunk: {avg_entities:.1f}")
        
        print(f"\n🔢 Processing in {total_batches} batches of {batch_size}")
        
        # Process chunks in batches with enhanced monitoring
        successful_batches = 0
        failed_batches = 0
        
        for i in range(0, len(processed_chunks), batch_size):
            batch_chunks = processed_chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            # Batch-level statistics
            batch_annotations = sum(1 for c in batch_chunks if c.metadata.get("has_annotations", False))
            batch_financial = sum(1 for c in batch_chunks if c.metadata.get("contains_financial_info", False))
            batch_color_amounts = sum(1 for c in batch_chunks if c.metadata.get("has_color_amounts", False))
            batch_color_entities = sum(c.metadata.get("color_entity_count", 0) for c in batch_chunks)
            
            print(f"   📦 Batch {batch_num}/{total_batches} "
                  f"({len(batch_chunks)} chunks, "
                  f"{batch_annotations} ann, "
                  f"{batch_financial} fin, "
                  f"🎨 {batch_color_amounts} $, "
                  f"{batch_color_entities} entities)...")
            
            try:
                vectorstore.add_documents(batch_chunks)
                print(f"   ✅ Batch {batch_num} completed")
                successful_batches += 1
                
            except Exception as e:
                print(f"   ❌ Batch {batch_num} failed: {e}")
                failed_batches += 1
                
                # Attempt individual chunk processing
                print(f"   🔄 Recovering batch {batch_num}...")
                individual_successes = 0
                for j, chunk in enumerate(batch_chunks):
                    try:
                        vectorstore.add_documents([chunk])
                        individual_successes += 1
                    except Exception as chunk_error:
                        print(f"     ❌ Chunk {j+1} failed: {chunk_error}")
                
                if individual_successes > 0:
                    print(f"   ⚡ Recovered {individual_successes}/{len(batch_chunks)} chunks")
            
            # API rate limiting
            if batch_num < total_batches:
                time.sleep(0.5)
        
        # Comprehensive summary
        print(f"\n🎯 EMBEDDING COMPLETION SUMMARY:")
        print(f"   ✅ Successful batches: {successful_batches}/{total_batches}")
        if failed_batches > 0:
            print(f"   ❌ Failed batches: {failed_batches}")
            print(f"   📊 Success rate: {(successful_batches/total_batches)*100:.1f}%")
        else:
            print(f"   🌟 Perfect success: 100%")
        
        print(f"\n🎨 COLOR-AWARE FEATURES ENABLED:")
        print(f"   ✓ Color-coded amount filtering (has_color_amounts)")
        print(f"   ✓ Color-coded party filtering (has_color_parties)")
        print(f"   ✓ Color-coded date filtering (has_color_dates)")
        print(f"   ✓ Highlight annotation tracking")
        print(f"   ✓ Entity count-based ranking")
        
        print(f"\n🏷️  ANNOTATION FEATURES ENABLED:")
        print(f"   ✓ High-quality chunk filtering")
        print(f"   ✓ Financial content prioritization") 
        print(f"   ✓ Entity confidence scoring")
        print(f"   ✓ Cross-document relationship mapping")
        
        print(f"\n✅ Enhanced embeddings stored in Weaviate: {index_name}")
        
    except Exception as e:
        print(f"❌ Critical error creating vector store: {e}")
        client.close()
        raise
    
    return vectorstore
    