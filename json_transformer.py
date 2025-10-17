from langchain_core.documents import Document
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import os


def transform_to_structured_json(documents: List[Document]) -> List[Document]:
    """
    Color-aware JSON transformation - prioritizes color metadata over regex
    
    Extraction priority:
    1. Color-coded metadata (human-verified, confidence: 0.95)
    2. Regex patterns (automated fallback, confidence: 0.75)
    """
    print("🔄 Starting color-aware JSON transformation...")
    transformed_documents = []
    
    # Track extraction statistics
    stats = {"total": 0, "color_used": 0, "regex_fallback": 0}
    
    for doc in documents:
        try:
            # Check if color metadata exists
            has_color_data = bool(doc.metadata.get("color_categories"))
            
            # Extract enhanced metadata (color-aware)
            enhanced_metadata = extract_enhanced_metadata(doc, has_color_data)
            
            # Extract document structure  
            content_hierarchy = extract_content_hierarchy(doc.page_content)
            
            # Extract special elements (color-aware)
            special_elements = extract_special_elements(doc, has_color_data)
            
            # Create structured JSON
            structured_data = {
                "document_metadata": enhanced_metadata,
                "content_hierarchy": content_hierarchy,
                "special_elements": special_elements,
                "color_metadata_used": has_color_data
            }
            
            # Create enhanced document
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,  # Preserve color metadata!
                    **enhanced_metadata,
                    "structured_data": json.dumps(structured_data, indent=2),
                    "transformation_timestamp": datetime.now().isoformat()
                }
            )
            
            transformed_documents.append(enhanced_doc)
            
            # Track stats
            stats["total"] += 1
            if has_color_data:
                stats["color_used"] += 1
            else:
                stats["regex_fallback"] += 1
            
        except Exception as e:
            print(f"⚠️  Transformation failed: {e}")
            transformed_documents.append(doc)
    
    print(f"✅ JSON transformation completed")
    print(f"   🎨 Color metadata used: {stats['color_used']}/{stats['total']}")
    print(f"   🔤 Regex fallback: {stats['regex_fallback']}/{stats['total']}")
    
    return transformed_documents


def extract_enhanced_metadata(doc: Document, has_color_data: bool) -> Dict[str, Any]:
    """Extract metadata prioritizing color-coded entities"""
    content = doc.page_content
    filename = os.path.basename(doc.metadata.get('source', ''))
    
    # Get color categories if available
    color_categories = doc.metadata.get("color_categories", {})
    
    # Parse if it's a JSON string
    if isinstance(color_categories, str):
        try:
            color_categories = json.loads(color_categories)
        except:
            color_categories = {}
    
    # Extract various metadata elements
    title = extract_document_title(content, filename)
    
    # 🎨 COLOR-AWARE: Parties extraction
    parties = extract_parties_color_aware(content, color_categories, has_color_data)
    
    # 🎨 COLOR-AWARE: Dates extraction
    dates = extract_dates_color_aware(content, color_categories, has_color_data)
    
    # 🎨 COLOR-AWARE: Purchase price extraction
    purchase_price = extract_purchase_price_color_aware(content, color_categories, has_color_data)
    
    doc_type = determine_document_type(content.lower())
    companies = extract_companies(content)
    
    return {
        "document_title": title,
        "document_type": doc_type,
        "parties": parties,
        "companies": companies,
        "dates": dates,
        "purchase_price": purchase_price,
        "page_number": doc.metadata.get('page', 1),
        "extraction_method": doc.metadata.get('extraction_method', 'unknown'),
        "filename": filename,
        "content_length": len(content),
        "legal_sections_count": count_legal_sections(content),
        "color_extraction_used": has_color_data
    }


def extract_parties_color_aware(content: str, color_categories: Dict, has_color: bool) -> Dict[str, List[str]]:
    """Extract parties prioritizing color-coded PARTY entities"""
    parties = {"buyers": [], "sellers": [], "other_parties": []}
    
    # 🎨 PRIORITY 1: Use color-marked parties (HIGHEST CONFIDENCE)
    if has_color and "PARTY" in color_categories:
        color_parties = color_categories["PARTY"]
        
        # Classify color parties as buyer/seller based on context
        for party in color_parties:
            party_lower = party.lower()
            
            # Check context around the party name
            party_context = ""
            pos = content.lower().find(party_lower)
            if pos != -1:
                context_start = max(0, pos - 100)
                context_end = min(len(content), pos + len(party) + 100)
                party_context = content[context_start:context_end].lower()
            
            # Classify based on context
            if any(term in party_context for term in ["buyer", "purchaser", "acquiring"]):
                parties["buyers"].append(party)
            elif any(term in party_context for term in ["seller", "vendor", "target"]):
                parties["sellers"].append(party)
            else:
                parties["other_parties"].append(party)
    
    # PRIORITY 2: Regex fallback (only if color missed something)
    if not has_color or len(parties["buyers"]) == 0:
        buyer_patterns = [
            r'(?:Buyer|Purchaser)(?:.*?)([A-Z][A-Za-z\s&,.]+(?:Inc\.|LLC|Corp\.|Company))',
        ]
        for pattern in buyer_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match.strip() not in parties["buyers"]:
                    parties["buyers"].append(match.strip())
    
    if not has_color or len(parties["sellers"]) == 0:
        seller_patterns = [
            r'(?:Seller|Vendor|Target)(?:.*?)([A-Z][A-Za-z\s&,.]+(?:Inc\.|LLC|Corp\.|Company))',
        ]
        for pattern in seller_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match.strip() not in parties["sellers"]:
                    parties["sellers"].append(match.strip())
    
    return parties


def extract_dates_color_aware(content: str, color_categories: Dict, has_color: bool) -> Dict[str, List[str]]:
    """Extract dates prioritizing color-coded DATE entities"""
    dates = {"execution_dates": [], "closing_dates": [], "other_dates": []}
    
    # 🎨 PRIORITY 1: Use color-marked dates
    if has_color and "DATE" in color_categories:
        color_dates = color_categories["DATE"]
        
        for date in color_dates:
            # Try to classify date type based on surrounding context
            date_context = ""
            pos = content.lower().find(date.lower())
            if pos != -1:
                context_start = max(0, pos - 50)
                context_end = min(len(content), pos + len(date) + 50)
                date_context = content[context_start:context_end].lower()
            
            if any(term in date_context for term in ["executed", "signed", "entered into"]):
                dates["execution_dates"].append(date)
            elif any(term in date_context for term in ["closing", "completion"]):
                dates["closing_dates"].append(date)
            else:
                dates["other_dates"].append(date)
    
    # PRIORITY 2: Regex fallback (only if no color dates found)
    if not has_color or len(dates["other_dates"]) == 0:
        date_patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            dates["other_dates"].extend(matches)
    
    return dates


def extract_purchase_price_color_aware(content: str, color_categories: Dict, has_color: bool) -> Optional[str]:
    """Extract purchase price prioritizing color-coded AMOUNT entities"""
    
    # 🎨 PRIORITY 1: Use color-marked amounts
    if has_color and "AMOUNT" in color_categories:
        color_amounts = color_categories["AMOUNT"]
        
        # Find the amount most likely to be purchase price
        for amount in color_amounts:
            # Check if this amount is near "purchase price" keywords
            amount_lower = amount.lower()
            pos = content.lower().find(amount_lower)
            if pos != -1:
                context_start = max(0, pos - 100)
                context_end = min(len(content), pos + len(amount) + 100)
                context = content[context_start:context_end].lower()
                
                if any(term in context for term in ["purchase price", "consideration", "aggregate", "total"]):
                    return amount
        
        # If no clear purchase price, return largest amount
        if color_amounts:
            return max(color_amounts, key=lambda x: len(x))
    
    # PRIORITY 2: Regex fallback
    price_patterns = [
        r'(?:purchase price|consideration)(?:.*?)\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion))?',
        r'aggregate.*?\$[\d,]+',
        r'upfront.*?\$[\d,]+'
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None


def extract_special_elements(doc: Document, has_color: bool) -> Dict[str, Any]:
    """Extract special elements using color metadata when available"""
    content = doc.page_content
    color_categories = doc.metadata.get("color_categories", {})
    
    if isinstance(color_categories, str):
        try:
            color_categories = json.loads(color_categories)
        except:
            color_categories = {}
    
    special_elements = {
        "tables": [],
        "signatures": [],
        "exhibits": [],
        "definitions": [],
        "dollar_amounts": [],
        "percentages": [],
        "cross_references": []
    }
    
    # 🎨 Use color metadata when available
    if has_color:
        special_elements["dollar_amounts"] = color_categories.get("AMOUNT", [])
        special_elements["percentages"] = color_categories.get("PERCENT", [])
        special_elements["cross_references"] = color_categories.get("CROSSREF", [])
        special_elements["definitions"] = color_categories.get("DEFINED_TERM", [])
    else:
        # Regex fallback
        special_elements["dollar_amounts"] = re.findall(
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?', 
            content, re.IGNORECASE
        )
        special_elements["percentages"] = re.findall(r'\d+(?:\.\d+)?%', content)
    
    # Extract signatures (no color equivalent)
    signature_pattern = r'(?:BY:|By:)\s*\n\s*.*?\n\s*(?:Name:|Title:|Its:)'
    special_elements["signatures"] = re.findall(signature_pattern, content, re.MULTILINE | re.IGNORECASE)
    
    # Exhibits
    if has_color and "CROSSREF" in color_categories:
        special_elements["exhibits"] = [
            ref for ref in color_categories["CROSSREF"] 
            if any(term in ref.lower() for term in ["exhibit", "schedule"])
        ]
    else:
        special_elements["exhibits"] = re.findall(
            r'(?:Exhibit|Schedule)\s+([A-Z]|\d+)', 
            content, re.IGNORECASE
        )
    
    return special_elements


def extract_content_hierarchy(content: str) -> List[Dict[str, Any]]:
    """Extract hierarchical structure (unchanged)"""
    hierarchy = []
    
    # Articles
    article_pattern = r'(?:^|\n)\s*ARTICLE\s+([IVXLCDM]+)\s*[.\-]?\s*(.*?)(?=\n|$)'
    for article in re.finditer(article_pattern, content, re.MULTILINE | re.IGNORECASE):
        hierarchy.append({
            "type": "article",
            "number": article.group(1),
            "title": article.group(2).strip(),
            "position": article.start(),
            "level": 1
        })
    
    # Sections
    section_pattern = r'(?:^|\n)\s*(?:SECTION|Section)\s+(\d+(?:\.\d+)*)\s*[.\-]?\s*(.*?)(?=\n|$)'
    for section in re.finditer(section_pattern, content, re.MULTILINE | re.IGNORECASE):
        hierarchy.append({
            "type": "section", 
            "number": section.group(1),
            "title": section.group(2).strip(),
            "position": section.start(),
            "level": 2
        })
    
    return sorted(hierarchy, key=lambda x: x['position'])


def extract_document_title(content: str, filename: str) -> str:
    """Extract document title (unchanged)"""
    lines = content.split('\n')[:10]
    for line in lines:
        line = line.strip()
        if 10 < len(line) < 200:
            if any(kw in line.lower() for kw in ['agreement', 'purchase', 'merger', 'acquisition']):
                return line
    return filename.replace('.pdf', '').replace('_', ' ').title()


def determine_document_type(content: str) -> str:
    """Determine document type (unchanged)"""
    if any(t in content for t in ['stock purchase agreement', 'share purchase']):
        return "Stock Purchase Agreement"
    elif any(t in content for t in ['asset purchase', 'asset acquisition']):
        return "Asset Purchase Agreement"
    elif 'merger agreement' in content:
        return "Merger Agreement"
    else:
        return "Purchase Agreement"


def extract_companies(content: str) -> List[str]:
    """Extract company names (unchanged)"""
    pattern = r'([A-Z][A-Za-z\s&,.]+(?:Inc\.|LLC|Corp\.|Corporation|Company|Ltd\.))'
    companies = re.findall(pattern, content)
    return list(set([c.strip() for c in companies if len(c.strip()) > 5]))


def count_legal_sections(content: str) -> int:
    """Count legal sections (unchanged)"""
    pattern = r'(?:^|\n)\s*(?:ARTICLE|SECTION|Section)\s+'
    return len(re.findall(pattern, content, re.MULTILINE | re.IGNORECASE))
