"""
batch_query_test.py
Automated batch testing script for RAG pipeline using all_queries.txt
"""

from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from retrieval_and_generation import retrieve_and_generate
import weaviate
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def parse_queries_from_file(filepath):
    """
    Parse all_queries.txt and extract structured queries
    
    Returns:
        List of dicts with structure:
        {
            'doc_number': 1,
            'doc_name': 'Integrated Brands SPA',
            'query_number': 1,
            'query_text': '...'
        }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    queries = []
    current_doc_number = None
    current_doc_name = None
    
    # Split by document sections
    doc_sections = re.split(r'##\s+\*\*(\d+)\.?\s*([^*]+)\*\*', content)
    
    # Process each document section (skip first empty element)
    for i in range(1, len(doc_sections), 3):
        doc_number = int(doc_sections[i])
        doc_name = doc_sections[i+1].strip()
        section_content = doc_sections[i+2] if i+2 < len(doc_sections) else ""
        
        # Extract queries from this section
        # Handle both "1." and "- " formats
        query_lines = []
        for line in section_content.split('\n'):
            line = line.strip()
            # Match numbered queries (1., 2., 3., etc.) or bullet points (-)
            if re.match(r'^(\d+\.|-)', line):
                # Remove leading number/bullet and whitespace
                query_text = re.sub(r'^(\d+\.|-)(\s*)', '', line).strip()
                if query_text:  # Only add non-empty queries
                    query_lines.append(query_text)
        
        # Add each query with metadata
        for query_num, query_text in enumerate(query_lines, 1):
            queries.append({
                'doc_number': doc_number,
                'doc_name': doc_name,
                'query_number': query_num,
                'query_text': query_text,
                'global_id': len(queries) + 1  # Overall query number
            })
    
    return queries


def connect_to_vectorstore():
    """Connect to existing Weaviate vectorstore"""
    print("Connecting to Weaviate vectorstore...")
    client = weaviate.connect_to_local(host="localhost", port=8081)
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


def run_batch_test(queries_file="all_queries.txt", output_format="json"):
    """
    Run batch testing on all queries and save results
    
    Args:
        queries_file: Path to queries text file
        output_format: "json" or "txt" for output format
    """
    client = None
    results = []
    
    try:
        # Parse queries
        print("\n" + "="*70)
        print("PARSING QUERIES FROM FILE")
        print("="*70)
        queries = parse_queries_from_file(queries_file)
        print(f"✅ Parsed {len(queries)} queries from {len(set(q['doc_number'] for q in queries))} documents")
        
        # Connect to vectorstore
        print("\n" + "="*70)
        print("CONNECTING TO RAG SYSTEM")
        print("="*70)
        vectorstore, client = connect_to_vectorstore()
        rag_pipeline = retrieve_and_generate(vectorstore)
        print("✅ RAG pipeline ready")
        
        # Run queries
        print("\n" + "="*70)
        print("RUNNING BATCH QUERIES")
        print("="*70)
        
        for idx, query_data in enumerate(queries, 1):
            doc_num = query_data['doc_number']
            doc_name = query_data['doc_name']
            query_num = query_data['query_number']
            query_text = query_data['query_text']
            
            print(f"\n[{idx}/{len(queries)}] Doc {doc_num}.{query_num}: {doc_name}")
            print(f"Query: {query_text[:80]}..." if len(query_text) > 80 else f"Query: {query_text}")
            
            try:
                # Get answer from RAG pipeline
                answer = rag_pipeline.invoke(query_text)
                
                # Store result
                result = {
                    'global_id': idx,
                    'doc_number': doc_num,
                    'doc_name': doc_name,
                    'query_number': query_num,
                    'query_text': query_text,
                    'answer': answer,
                    'status': 'success',
                    'answer_length': len(answer),
                    'has_financial_data': '$' in answer or 'million' in answer.lower() or 'billion' in answer.lower()
                }
                
                print(f"✅ Answer length: {len(answer)} chars")
                
            except Exception as e:
                result = {
                    'global_id': idx,
                    'doc_number': doc_num,
                    'doc_name': doc_name,
                    'query_number': query_num,
                    'query_text': query_text,
                    'answer': None,
                    'status': 'error',
                    'error': str(e)
                }
                print(f"❌ Error: {e}")
            
            results.append(result)
        
        # Save results
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "json":
            output_file = f"batch_test_results_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {output_file}")
        
        else:  # txt format
            output_file = f"batch_test_results_{timestamp}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("RAG PIPELINE BATCH TEST RESULTS\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Queries: {len(results)}\n")
                successful = sum(1 for r in results if r['status'] == 'success')
                f.write(f"Successful: {successful}/{len(results)}\n")
                f.write("="*80 + "\n\n")
                
                for result in results:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"QUERY #{result['global_id']}\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Document: [{result['doc_number']}] {result['doc_name']}\n")
                    f.write(f"Query Number: {result['query_number']}\n")
                    f.write(f"\nQUESTION:\n{result['query_text']}\n")
                    f.write(f"\n{'-'*80}\n")
                    
                    if result['status'] == 'success':
                        f.write(f"ANSWER:\n{result['answer']}\n")
                        f.write(f"\n{'-'*80}\n")
                        f.write(f"Status: ✅ Success\n")
                        f.write(f"Answer Length: {result['answer_length']} characters\n")
                        f.write(f"Contains Financial Data: {'Yes' if result['has_financial_data'] else 'No'}\n")
                    else:
                        f.write(f"ERROR: {result.get('error', 'Unknown error')}\n")
                        f.write(f"Status: ❌ Failed\n")
                    
                    f.write(f"{'='*80}\n")
            
            print(f"✅ Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        print(f"Total Queries: {len(results)}")
        print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
        print(f"Failed: {failed} ({failed/len(results)*100:.1f}%)")
        
        if successful > 0:
            avg_length = sum(r['answer_length'] for r in results if r['status'] == 'success') / successful
            with_financial = sum(1 for r in results if r['status'] == 'success' and r['has_financial_data'])
            print(f"Average Answer Length: {avg_length:.0f} characters")
            print(f"Answers with Financial Data: {with_financial}/{successful} ({with_financial/successful*100:.1f}%)")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if client:
            client.close()
            print("\n✅ Connection closed")


if __name__ == "__main__":
    import sys
    
    # Default to JSON output, can specify 'txt' as command line argument
    output_format = sys.argv[1] if len(sys.argv) > 1 else "json"
    
    if output_format not in ["json", "txt"]:
        print("Usage: python batch_query_test.py [json|txt]")
        print("Defaulting to JSON format...")
        output_format = "json"
    
    run_batch_test(queries_file="all_queries.txt", output_format=output_format)
