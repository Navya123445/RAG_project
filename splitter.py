from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import json


def text_splitter(documents: List[Document]) -> List[Document]:
    """
    Enhanced legal document splitter with:
    - JSON metadata preservation
    - Annotation data preservation
    - COLOR-AWARE entity boundary respect
    - Intelligent chunk metadata
    """
    print("🎨 Splitting documents with color-aware chunking...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=800,
        separators=[
            "\n\nARTICLE ",
            "\n\nSECTION ",
            "\n\nSection ",
            "\n\n",
            "\n",
            ". ",
            " ",
        ]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Process chunks with color metadata
    enhanced_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Add basic chunk metadata
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        # ENHANCEMENT 1: Process structured JSON data
        if "structured_data" in chunk.metadata:
            try:
                structured_data = json.loads(chunk.metadata["structured_data"])
                doc_meta = structured_data.get("document_metadata", {})
                
                chunk.metadata["document_type"] = doc_meta.get("document_type", "Unknown")
                chunk.metadata["parties"] = str(doc_meta.get("parties", {}))
                chunk.metadata["purchase_price"] = doc_meta.get("purchase_price", "")
            except json.JSONDecodeError:
                pass
        
        # ENHANCEMENT 2: Process annotation data
        if "annotations" in chunk.metadata:
            try:
                annotations = json.loads(chunk.metadata["annotations"])
                financial_info = annotations.get("financial_information", {})
                entities = annotations.get("legal_entities", {})
                confidence_scores = annotations.get("confidence_scores", {})
                
                chunk.metadata["has_annotations"] = True
                chunk.metadata["annotation_confidence"] = confidence_scores.get("overall_confidence", 0)
                chunk.metadata["financial_confidence"] = confidence_scores.get("financial_confidence", 0)
                chunk.metadata["entity_confidence"] = confidence_scores.get("entity_confidence", 0)
                
                chunk.metadata["company_count"] = len(entities.get("companies", []))
                chunk.metadata["person_count"] = len(entities.get("persons", []))
                
                monetary_amounts = financial_info.get("monetary_amounts", [])
                chunk.metadata["financial_amount_count"] = len(monetary_amounts)
                
                if monetary_amounts:
                    top_amounts = []
                    for amt_info in monetary_amounts[:3]:
                        amount = amt_info.get("amount", "")
                        confidence = amt_info.get("confidence", 0)
                        if confidence > 0.7:
                            top_amounts.append(amount)
                    if top_amounts:
                        chunk.metadata["key_financial_amounts"] = "|".join(top_amounts)
                
                chunk.metadata["high_quality_chunk"] = confidence_scores.get("overall_confidence", 0) > 0.8
            except json.JSONDecodeError:
                chunk.metadata["has_annotations"] = False
                chunk.metadata["high_quality_chunk"] = False
        else:
            chunk.metadata["has_annotations"] = False
            chunk.metadata["high_quality_chunk"] = False
        
        # 🎨 ENHANCEMENT 3: Process COLOR METADATA
        if "color_entities" in chunk.metadata:
            # Find which color-coded entities are in this chunk
            chunk_color_entities = []
            chunk_color_categories = {}
            
            for entity in chunk.metadata.get("color_entities", []):
                if entity["text"] in chunk.page_content:
                    chunk_color_entities.append(entity)
                    category = entity["category"]
                    if category not in chunk_color_categories:
                        chunk_color_categories[category] = []
                    chunk_color_categories[category].append(entity["text"])
            
            # Add color metadata to chunk
            chunk.metadata["chunk_color_entities"] = str(chunk_color_entities)  # String for Weaviate
            chunk.metadata["chunk_color_categories"] = str(chunk_color_categories)
            
            # Add boolean flags for filtering
            chunk.metadata["has_color_amounts"] = "AMOUNT" in chunk_color_categories
            chunk.metadata["has_color_parties"] = "PARTY" in chunk_color_categories
            chunk.metadata["has_color_dates"] = "DATE" in chunk_color_categories
            chunk.metadata["has_color_qualifiers"] = "QUALIFIER" in chunk_color_categories
            chunk.metadata["has_color_percentages"] = "PERCENT" in chunk_color_categories
            chunk.metadata["has_color_crossrefs"] = "CROSSREF" in chunk_color_categories
            
            # Count color entities
            chunk.metadata["color_entity_count"] = len(chunk_color_entities)
        else:
            chunk.metadata["has_color_amounts"] = False
            chunk.metadata["has_color_parties"] = False
            chunk.metadata["color_entity_count"] = 0
        
        # Process highlighted annotations if present
        if "highlighted_annotations" in chunk.metadata:
            highlights_in_chunk = []
            for annot in chunk.metadata.get("highlighted_annotations", []):
                if annot["text"] in chunk.page_content:
                    highlights_in_chunk.append(annot)
            
            chunk.metadata["highlight_count"] = len(highlights_in_chunk)
            chunk.metadata["has_highlights"] = len(highlights_in_chunk) > 0
        else:
            chunk.metadata["has_highlights"] = False
        
        # ENHANCEMENT 4: Legal structure detection
        chunk_content = chunk.page_content.lower()
        
        has_financial_section = any(term in chunk_content for term in [
            "purchase price", "consideration", "payment", "milestone", "earnout",
            "royalty", "cash", "$", "million", "thousand"
        ])
        
        has_party_section = any(term in chunk_content for term in [
            "buyer", "seller", "purchaser", "vendor", "target", "acquirer"
        ])
        
        has_legal_refs = any(term in chunk_content for term in [
            "article", "section", "subsection", "exhibit", "schedule"
        ])
        
        chunk.metadata["contains_financial_info"] = has_financial_section
        chunk.metadata["contains_party_info"] = has_party_section
        chunk.metadata["contains_legal_refs"] = has_legal_refs
        
        # ENHANCEMENT 5: Calculate relevance score (including color data)
        relevance_score = 0
        
        # Boost for annotations
        if chunk.metadata.get("has_annotations", False):
            relevance_score += chunk.metadata.get("annotation_confidence", 0) * 0.3
        
        # Boost for color-coded entities (NEW)
        if chunk.metadata.get("color_entity_count", 0) > 0:
            relevance_score += 0.25  # Strong boost for color-coded content
        
        # Extra boost for color-coded amounts
        if chunk.metadata.get("has_color_amounts", False):
            relevance_score += 0.2
        
        # Boost for financial content
        if has_financial_section:
            relevance_score += 0.15
        
        # Boost for party information
        if has_party_section:
            relevance_score += 0.1
        
        chunk.metadata["relevance_score"] = min(1.0, relevance_score)
        
        enhanced_chunks.append(chunk)
    
    # Print statistics
    print(f"✅ Documents split into {len(enhanced_chunks)} color-aware chunks")
    print(f"📊 Chunks with annotations: {sum(1 for c in enhanced_chunks if c.metadata.get('has_annotations', False))}")
    print(f"🎨 Chunks with color entities: {sum(1 for c in enhanced_chunks if c.metadata.get('color_entity_count', 0) > 0)}")
    print(f"💰 Chunks with color-coded amounts: {sum(1 for c in enhanced_chunks if c.metadata.get('has_color_amounts', False))}")
    print(f"👥 Chunks with color-coded parties: {sum(1 for c in enhanced_chunks if c.metadata.get('has_color_parties', False))}")
    print(f"💰 Chunks with financial info: {sum(1 for c in enhanced_chunks if c.metadata.get('contains_financial_info', False))}")
    
    return enhanced_chunks
