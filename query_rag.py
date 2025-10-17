from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from retrieval_and_generation import retrieve_and_generate
import weaviate
import os
from dotenv import load_dotenv

load_dotenv()

def connect_to_vectorstore():
    """Connect to existing Weaviate vectorstore"""
    print("Connecting to Weaviate vectorstore...")
    
    client = weaviate.connect_to_local(host="localhost", port=8080)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = os.getenv("WEAVIATE_INDEX_NAME", "LegalDocuments")
    
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name=index_name,
        text_key="text",
        embedding=embeddings
    )
    
    print(f"Connected to vectorstore: {index_name}")
    return vectorstore, client

if __name__ == "__main__":
    client = None
    try:
        # Connect to vectorstore
        vectorstore, client = connect_to_vectorstore()
        
        # Create RAG pipeline
        rag_pipeline = retrieve_and_generate(vectorstore)
        
        # Simple test queries
        test_queries = [
            # Purchase-price questions
            "What is the purchase price for Heat Biologics shares?",
            "State the aggregate consideration in the Anaren Holdings SPA.",
            # Party-identification
            "Who are the buyer and seller in the Ariba Inc. acquisition?",
            "List all parties in the Pacific Architects and Engineers deal.",
            # Closing conditions
            "Summarize the closing conditions in the Meta Platforms agreement.",
            "Which regulatory approvals are required in the ISP Optics SPA?",
            # Cash / stock breakdown
            "What is the cash consideration in the Q2 Software transaction?",
            "How much stock is issued in the Oragenics deal?",
            # Indemnification & caps
            "Describe the indemnification cap in the Pilgrim's Pride agreement.",
            "What escrow amount is held back in the Broder Bros. SPA?",
            # Reps & warranties
            "Give key representations and warranties in the CAI International SPA."
        ]
        
        print("\n" + "="*60)
        print("RAG SYSTEM TEST")
        print("="*60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/5] Query: {query}")
            print("-" * 60)
            
            try:
                answer = rag_pipeline.invoke(query)
                print(f"Answer: {answer}")
                
                # Simple quality check
                has_details = '$' in answer or 'million' in answer.lower() or len(answer) > 300
                print(f"Quality: {'✅ Detailed' if has_details else '⚠️  Basic'}")
                
            except Exception as e:
                print(f"Error: {e}")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        if client:
            client.close()
