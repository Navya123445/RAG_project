from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import json
from dotenv import load_dotenv

load_dotenv()

def retrieve_and_generate(vectorstore):
    model_name = os.getenv("OPENAI_MODEL", "gpt-5")
    llm = ChatOpenAI(model=model_name, temperature=0.01)
    
    def is_complex_financial_query(question):
        """Determine if query requires more iterations for complex financial analysis"""
        complex_indicators = [
            'purchase price', 'consideration', 'milestone', 'earnout', 'payment structure',
            'royalty', 'contingent', 'closing consideration', 'aggregate', 'valuation',
            'compare', 'across all', 'multiple', 'pattern', 'frequently'
        ]
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in complex_indicators)
    
    def recursive_retrieval(question, max_iterations=None):
        """
        Enhanced adaptive recursive retrieval with metadata filtering:
        1. Initial broad search with smart metadata filtering
        2. Analyze what's missing  
        3. Generate targeted follow-up queries with specific filters
        4. Retrieve missing information using enhanced context
        5. Combine all contexts with structured metadata
        """
        
        # ENHANCEMENT: Dynamic iteration count based on query complexity
        if max_iterations is None:
            max_iterations = 5 if is_complex_financial_query(question) else 3
        
        all_contexts = []
        current_question = question
        
        # ENHANCEMENT 1: Smart initial metadata filtering
        metadata_filters = extract_query_filters(question, llm)
        
        for iteration in range(max_iterations):
            print(f"  🔍 Retrieval iteration {iteration + 1}: {current_question[:100]}...")
            
            # ENHANCEMENT 2: Improved retrieval coverage with annotation awareness
            if metadata_filters and iteration == 0:
                print(f"    📋 Applying filters: {metadata_filters}")
                docs = vectorstore.similarity_search(
                    current_question, 
                    k=20,  # Increased from 12
                    filter=metadata_filters
                )
            else:
                # ENHANCEMENT: Progressive increase in retrieval coverage
                k_value = 25 + (iteration * 5)  # 25, 30, 35, 40, 45 for iterations 0-4
                docs = vectorstore.similarity_search(current_question, k=k_value)
            
            if docs:
                all_contexts.extend(docs)
                
                # If this is not the final iteration, analyze what might be missing
                if iteration < max_iterations - 1:
                    # ENHANCEMENT 3: Enhanced context analysis with metadata AND annotations
                    context_summary = format_enhanced_context(docs[:3])
                    
                    analysis_prompt = f"""
                    Original Question: {question}
                    
                    Current Retrieved Context with Metadata and Annotations:
                    {context_summary}
                    
                    Based on the original question and the structured context above, what specific information is still missing? 
                    Consider:
                    - Different document types or sections
                    - Specific parties, dates, or financial details (upfront payments, milestones, royalties)
                    - Cross-references between documents
                    - Alternative terminology (consideration vs purchase price vs transaction value)
                    - Annotation data showing financial patterns or entity relationships
                    
                    Generate a focused follow-up search query (max 15 words) to find the missing information.
                    If the context seems complete, respond with "COMPLETE".
                    
                    Follow-up query:"""
                    
                    follow_up = llm.invoke(analysis_prompt).content.strip()
                    
                    if follow_up == "COMPLETE" or len(follow_up.split()) > 15:
                        break
                    else:
                        current_question = follow_up
                else:
                    break
            else:
                break
        
        # ENHANCEMENT 4: Deduplicate with metadata awareness
        unique_contexts = []
        seen_content = set()
        
        for doc in all_contexts:
            # Create fingerprint from content + source + page
            content_fingerprint = f"{doc.page_content[:150]}_{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
            if content_fingerprint not in seen_content:
                unique_contexts.append(doc)
                seen_content.add(content_fingerprint)
        
        # ENHANCEMENT: Return more contexts for complex queries
        context_limit = 35 if is_complex_financial_query(question) else 25
        return unique_contexts[:context_limit]
    
    # ENHANCEMENT 5: Updated professional prompt with enhanced financial focus AND annotation awareness
    template = """You are a senior legal analyst providing precise analysis of Stock Purchase Agreements and legal documents.

COMPREHENSIVE CONTEXT WITH STRUCTURED METADATA AND ANNOTATIONS:
{enhanced_context}

ORIGINAL QUESTION: {question}

INSTRUCTIONS FOR ENHANCED LEGAL ANALYSIS:
1. Leverage the structured metadata (document types, parties, dates, purchase prices) AND annotation data for precise answers
2. Cross-reference information across multiple documents when available
3. Extract ALL financial amounts including: upfront payments, closing consideration, milestone payments, earnouts, royalties, stock consideration
4. 🎨 PRIORITIZE COLOR-CODED ENTITIES:
   - Yellow-highlighted text = Dollar amounts (HIGHEST PRIORITY for financial queries)
   - Blue-highlighted text = Party names (HIGHEST PRIORITY for party queries)
   - Green-highlighted text = Percentages
   - Pink-highlighted text = Defined terms
   - Brown-highlighted text = Cross-references
   - Purple-highlighted text = Qualifiers (material, knowledge, etc.)
5. Use annotation confidence scores to prioritize information - higher confidence annotations should be weighted more heavily
6. Identify payment structures: cash, stock, contingent payments, percentage-based payments
7. Quote EXACT language from documents when available, including dollar amounts and percentages
8. Cite document sources and page numbers from metadata
9. For complex payment structures, break down each component separately (upfront + milestones + royalties)
10. If annotation confidence is low (<0.7), flag this information as potentially requiring verification
11. If information is definitively absent after thorough analysis, state: "This information is not contained in the provided documents"
12. Use document hierarchy (Articles/Sections) from structured data when available
13. Give ACTIONABLE legal information suitable for professional use

CRITICAL: For financial queries, search for alternative terminology AND use annotation data:
- "Purchase price" may also appear as "consideration", "transaction value", "aggregate consideration"
- Look for "cash consideration", "stock consideration", "earnout payments", "milestone payments"
- Check annotation financial_information for extracted monetary amounts with confidence scores
- Use entity annotations for accurate party identification and role disambiguation
- Leverage legal reference annotations for proper section citations

STRUCTURE YOUR RESPONSE:
- Direct answer with ALL financial components found (prioritizing high-confidence annotations)
- Supporting evidence with exact quotes and document references  
- Breakdown of payment structure (upfront, contingent, percentage-based) with confidence indicators
- Cross-document analysis when multiple documents are relevant
- Specific section references from document hierarchy
- Annotation confidence summary for key findings
- Any limitations in the available information

ENHANCED LEGAL ANALYSIS:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # ENHANCEMENT 6: Enhanced RAG chain with structured context formatting
    rag_chain = (
        {
            "enhanced_context": lambda x: format_enhanced_context(recursive_retrieval(x)), 
            "question": RunnablePassthrough()
        }
        | prompt  
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def extract_query_filters(question, llm):
    """Extract appropriate metadata filters from user question"""
    
    filter_extraction_prompt = f"""
    Analyze this legal query and determine appropriate metadata filters for SPA document search:
    
    Query: {question}
    
    Available metadata fields:
    - document_type (e.g., "Stock Purchase Agreement", "Asset Purchase Agreement")
    - parties (buyer/seller names)
    - purchase_price (if question involves financial terms)
    - filename (specific document names)
    - has_annotations (true/false - prioritize annotated documents)
    - annotation_confidence (0.0-1.0 - prioritize high confidence)
    
    Respond ONLY with valid JSON filter object or "null" if no specific filters needed.
    Examples:
    {{"document_type": "Stock Purchase Agreement"}}
    {{"parties": "Nvidia"}}
    {{"has_annotations": true, "annotation_confidence": 0.8}}
    null
    
    Filter JSON:"""
    
    try:
        filter_response = llm.invoke(filter_extraction_prompt).content.strip()
        
        if filter_response.lower() == "null":
            return None
        
        # Parse and validate the filter
        filters = json.loads(filter_response)
        return filters if isinstance(filters, dict) else None
        
    except (json.JSONDecodeError, Exception):
        return None


def extract_query_filters(question, llm):
    """Extract appropriate metadata filters from user question (ENHANCED with color filters)"""
    filter_extraction_prompt = f"""
Analyze this legal query and determine appropriate metadata filters for SPA document search:

Query: {question}

Available metadata fields:
- document_type (e.g., "Stock Purchase Agreement")
- parties (buyer/seller names)
- purchase_price (if question involves financial terms)
- filename (specific document names)
- has_annotations (true/false - prioritize annotated documents)
- annotation_confidence (0.0-1.0)
- has_color_amounts (true/false - chunks with color-coded dollar amounts)
- has_color_parties (true/false - chunks with color-coded party names)
- has_color_dates (true/false - chunks with color-coded dates)
- has_color_percentages (true/false - chunks with color-coded percentages)
- has_color_qualifiers (true/false - chunks with color-coded qualifiers like "material")
- color_entity_count (number - more entities = higher priority)

🎨 COLOR FILTERS are especially useful for:
- Financial queries → use has_color_amounts=true
- Party-related queries → use has_color_parties=true  
- Date queries → use has_color_dates=true
- Percentage/ratio queries → use has_color_percentages=true

Respond ONLY with valid JSON filter object or "null" if no specific filters needed.

Examples:
{{"has_color_amounts": true, "contains_financial_info": true}}
{{"has_color_parties": true, "parties": "Nvidia"}}
{{"has_color_dates": true}}
{{"document_type": "Stock Purchase Agreement", "has_color_amounts": true}}
null

Filter JSON:"""

    try:
        filter_response = llm.invoke(filter_extraction_prompt).content.strip()
        if filter_response.lower() == "null":
            return None
        
        filters = json.loads(filter_response)
        return filters if isinstance(filters, dict) else None
        
    except (json.JSONDecodeError, Exception):
        return None



def format_enhanced_context(docs):
    """Format documents with rich metadata AND annotations for enhanced context"""
    if not docs:
        return "No relevant documents found."
    
    formatted_contexts = []
    
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        
        # Extract structured data if available
        structured_info = ""
        annotation_info = ""
        
        # ENHANCEMENT 1: Include structured JSON data
        structured_str = metadata.get("structured_data")
        if structured_str and structured_str not in ("", "null", "{}"):
            try:
                structured_data = json.loads(structured_str)
                doc_meta = structured_data.get("document_metadata", {})
                
                structured_info = f"""
📄 Document: {doc_meta.get('document_title', 'Unknown')}
📂 Type: {doc_meta.get('document_type', 'Unknown')}
👥 Parties: {doc_meta.get('parties', {})}
💰 Purchase Price: {doc_meta.get('purchase_price', 'Not specified')}
📅 Dates: {doc_meta.get('dates', {})}
📍 Page: {metadata.get('page', 'Unknown')}
🔧 Source: {metadata.get('filename', metadata.get('source', 'Unknown'))}
"""
            except (json.JSONDecodeError, KeyError):
                structured_info = f"📍 Page: {metadata.get('page', 'Unknown')} | 🔧 Source: {metadata.get('source', 'Unknown')}"
        else:
            structured_info = f"📍 Page: {metadata.get('page', 'Unknown')} | 🔧 Source: {metadata.get('source', 'Unknown')}"
        
        # ENHANCEMENT 2: Include annotation data for better retrieval
        annotations_str = metadata.get("annotations")
        if annotations_str and annotations_str not in ("", "null", "{}"):
            try:
                annotations = json.loads(annotations_str)
                                
                # Extract key annotation insights
                financial_amounts = annotations.get("financial_information", {}).get("monetary_amounts", [])
                entities = annotations.get("legal_entities", {})
                confidence_scores = annotations.get("confidence_scores", {})
                overall_confidence = confidence_scores.get("overall_confidence", 0)
                financial_confidence = confidence_scores.get("financial_confidence", 0)
                
                # Format annotation summary
                if financial_amounts:
                    top_amounts = []
                    for amt in financial_amounts[:3]:  # Top 3 amounts
                        amount_text = amt["amount"]
                        conf = amt.get("confidence", 0)
                        top_amounts.append(f"{amount_text} (conf: {conf:.2f})")
                    
                    annotation_info = f"""
🏷️  ANNOTATIONS:
💰 Key Amounts: {', '.join(top_amounts)}
🏢 Companies: {len(entities.get('companies', []))} found
👤 Persons: {len(entities.get('persons', []))} found
📊 Overall Confidence: {overall_confidence:.2f}
💵 Financial Confidence: {financial_confidence:.2f}
"""
                else:
                    annotation_info = f"""
🏷️  ANNOTATIONS:
🏢 Companies: {len(entities.get('companies', []))} found
👤 Persons: {len(entities.get('persons', []))} found  
📊 Overall Confidence: {overall_confidence:.2f}
💵 Financial Confidence: {financial_confidence:.2f}
"""
            except (json.JSONDecodeError, KeyError):
                annotation_info = ""
        
        context_block = f"""
=== DOCUMENT {i+1} ===
{structured_info}{annotation_info}

📝 Content:
{doc.page_content}

"""
        formatted_contexts.append(context_block)
    
    return "\n".join(formatted_contexts)
