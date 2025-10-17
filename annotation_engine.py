import spacy
import re
import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class LegalAnnotationEngine:
    """
    Color-Aware Legal Document Annotation Engine
    
    Combines:
    - Color-coded metadata (human-verified markups)
    - spaCy NER (entity recognition)
    - Regex patterns (structured extraction)
    """
    
    def __init__(self):
        print("🔧 Initializing Color-Aware Annotation Engine...")
        self.load_nlp_models()
        self.legal_patterns = self._define_legal_patterns()
        self.financial_patterns = self._define_financial_patterns()
        print("✅ Annotation Engine ready")
    
    def load_nlp_models(self):
        """Load spaCy with fallback"""
        try:
            self.nlp = spacy.load("en_core_web_lg")
            print("📚 Loaded en_core_web_lg")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_md")
                print("📚 Loaded en_core_web_md")
            except OSError:
                print("⚠️  No spaCy model found. Install: python -m spacy download en_core_web_md")
                self.nlp = None
    
    def annotate_documents(self, documents: List[Document]) -> List[Document]:
        """Annotate documents combining color metadata + NLP"""
        print(f"🏷️  Annotating {len(documents)} documents (color-aware)...")
        
        annotated_documents = []
        for i, doc in enumerate(documents):
            try:
                annotations = self._annotate_single_document(doc)
                annotated_doc = self._create_annotated_document(doc, annotations)
                annotated_documents.append(annotated_doc)
            except Exception as e:
                print(f"⚠️  Document {i+1} annotation failed: {e}")
                annotated_documents.append(doc)
        
        print(f"✅ Annotation completed")
        return annotated_documents
    
    def _annotate_single_document(self, doc: Document) -> Dict[str, Any]:
        """Comprehensive annotation with color integration"""
        content = doc.page_content
        metadata = doc.metadata
        
        # 🎨 INTEGRATION: Check if color metadata exists
        color_entities = metadata.get("color_entities", [])
        color_categories = metadata.get("color_categories", {})
        has_colors = len(color_entities) > 0
        
        # 1. Legal entities (spaCy NER + color PARTY markup)
        legal_entities = self._extract_legal_entities(content, color_entities)
        
        # 2. Financial info (prioritize color AMOUNT markup)
        financial_info = self._extract_financial_information(content, color_entities, has_colors)
        
        # 3. Legal references (combine color CROSSREF + regex)
        legal_references = self._extract_legal_references(content, color_entities)
        
        # 4. Dates (prioritize color DATE markup)
        dates_deadlines = self._extract_dates(content, color_entities)
        
        # 5. Confidence scores (boost for color-verified data)
        confidence_scores = self._calculate_confidence_scores(
            legal_entities, financial_info, legal_references, has_colors
        )
        
        return {
            "legal_entities": legal_entities,
            "financial_information": financial_info,
            "legal_references": legal_references,
            "dates_and_deadlines": dates_deadlines,
            "confidence_scores": confidence_scores,
            "color_integration_used": has_colors,
            "annotation_timestamp": datetime.now().isoformat()
        }
    
    def _extract_legal_entities(self, content: str, color_entities: List[Dict]) -> Dict[str, Any]:
        """Extract entities prioritizing color-marked PARTY entities"""
        entities = {"companies": [], "persons": [], "roles": []}
        
        # 🎨 PRIORITY 1: Color-marked parties (HIGHEST CONFIDENCE)
        color_parties = [e for e in color_entities if e.get("category") == "PARTY"]
        for party in color_parties:
            entities["companies"].append({
                "text": party["text"],
                "label": "PARTY",
                "confidence": 0.95,  # High confidence - human-verified
                "source": "color_markup"
            })
        
        # PRIORITY 2: spaCy NER
        if self.nlp:
            doc_nlp = self.nlp(content)
            for ent in doc_nlp.ents:
                if ent.label_ == "ORG":
                    # Check if already captured by color
                    if not any(ent.text.lower() in p["text"].lower() for p in entities["companies"]):
                        entities["companies"].append({
                            "text": ent.text,
                            "label": "ORG",
                            "confidence": 0.75,
                            "source": "spacy_ner"
                        })
                elif ent.label_ == "PERSON":
                    entities["persons"].append({
                        "text": ent.text,
                        "label": "PERSON",
                        "confidence": 0.7,
                        "source": "spacy_ner"
                    })
        
        # PRIORITY 3: Pattern-based extraction
        roles_pattern = r'\b(?:Buyer|Seller|Purchaser|Vendor|Target|Acquirer)\b'
        for match in re.finditer(roles_pattern, content, re.IGNORECASE):
            entities["roles"].append({
                "text": match.group(0),
                "label": "ROLE",
                "confidence": 0.85,
                "source": "regex_pattern"
            })
        
        return self._deduplicate_entities(entities)
    
    def _extract_financial_information(self, content: str, color_entities: List[Dict], has_colors: bool) -> Dict[str, Any]:
        """Extract financial info prioritizing color-marked AMOUNT entities"""
        financial_info = {
            "monetary_amounts": [],
            "percentages": [],
            "payment_structures": []
        }
        
        # 🎨 PRIORITY 1: Color-marked amounts (HIGHEST CONFIDENCE)
        color_amounts = [e for e in color_entities if e.get("category") == "AMOUNT"]
        for amount in color_amounts:
            # Get context around the amount
            pos = content.find(amount["text"])
            if pos != -1:
                context_start = max(0, pos - 50)
                context_end = min(len(content), pos + len(amount["text"]) + 50)
                context = content[context_start:context_end]
                
                financial_info["monetary_amounts"].append({
                    "amount": amount["text"],
                    "context": context.strip(),
                    "confidence": 0.95,  # High - human-verified via color
                    "source": "color_markup",
                    "color_category": "AMOUNT"
                })
        
        # PRIORITY 2: Regex extraction (for amounts missed by color)
        dollar_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?'
        for match in re.finditer(dollar_pattern, content, re.IGNORECASE):
            amount_text = match.group(0)
            
            # Skip if already captured by color markup
            if not any(amount_text in ca["text"] for ca in color_amounts):
                context_start = max(0, match.start() - 50)
                context_end = min(len(content), match.end() + 50)
                context = content[context_start:context_end]
                
                # Lower confidence - not verified by color
                confidence = 0.75 if has_colors else 0.8
                
                financial_info["monetary_amounts"].append({
                    "amount": amount_text,
                    "context": context.strip(),
                    "confidence": confidence,
                    "source": "regex_pattern",
                    "color_category": None
                })
        
        # 🎨 PRIORITY 1: Color-marked percentages
        color_percentages = [e for e in color_entities if e.get("category") == "PERCENT"]
        for pct in color_percentages:
            financial_info["percentages"].append({
                "percentage": pct["text"],
                "confidence": 0.95,
                "source": "color_markup"
            })
        
        # PRIORITY 2: Regex percentages (fallback)
        pct_pattern = r'\d+(?:\.\d+)?%'
        for match in re.finditer(pct_pattern, content):
            pct_text = match.group(0)
            if not any(pct_text in cp["text"] for cp in color_percentages):
                financial_info["percentages"].append({
                    "percentage": pct_text,
                    "confidence": 0.8,
                    "source": "regex_pattern"
                })
        
        # Payment structure detection
        payment_keywords = ["upfront", "milestone", "earnout", "royalty", "escrow"]
        for keyword in payment_keywords:
            pattern = rf'{keyword}.*?(?:\$[\d,]+|payment)'
            for match in re.finditer(pattern, content, re.IGNORECASE):
                financial_info["payment_structures"].append({
                    "type": keyword,
                    "text": match.group(0),
                    "confidence": 0.75
                })
        
        return financial_info
    
    def _extract_legal_references(self, content: str, color_entities: List[Dict]) -> Dict[str, Any]:
        """Extract legal references combining color CROSSREF + regex"""
        references = {"articles": [], "sections": [], "exhibits": []}
        
        # 🎨 Color-marked cross-references (HIGHEST CONFIDENCE)
        color_refs = [e for e in color_entities if e.get("category") == "CROSSREF"]
        for ref in color_refs:
            ref_text = ref["text"]
            if "article" in ref_text.lower():
                references["articles"].append({
                    "reference": ref_text,
                    "confidence": 0.95,
                    "source": "color_markup"
                })
            elif "section" in ref_text.lower():
                references["sections"].append({
                    "reference": ref_text,
                    "confidence": 0.95,
                    "source": "color_markup"
                })
            elif "exhibit" in ref_text.lower() or "schedule" in ref_text.lower():
                references["exhibits"].append({
                    "reference": ref_text,
                    "confidence": 0.95,
                    "source": "color_markup"
                })
        
        # Regex-based extraction (fallback)
        patterns = {
            "articles": r'ARTICLE\s+(?:[IVXLCDM]+|\d+)',
            "sections": r'(?:SECTION|Section)\s+\d+(?:\.\d+)*',
            "exhibits": r'(?:Exhibit|Schedule)\s+[A-Z\d]+'
        }
        
        for ref_type, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                ref_text = match.group(0)
                # Skip if already in color refs
                if not any(ref_text.lower() in cr["text"].lower() for cr in color_refs):
                    references[ref_type].append({
                        "reference": ref_text,
                        "confidence": 0.85,
                        "source": "regex_pattern"
                    })
        
        return references
    
    def _extract_dates(self, content: str, color_entities: List[Dict]) -> Dict[str, Any]:
        """Extract dates prioritizing color DATE markup"""
        dates_info = {"execution_dates": [], "closing_dates": [], "other_dates": []}
        
        # 🎨 Color-marked dates
        color_dates = [e for e in color_entities if e.get("category") == "DATE"]
        for date in color_dates:
            dates_info["other_dates"].append({
                "date": date["text"],
                "confidence": 0.95,
                "source": "color_markup"
            })
        
        # Context-based date extraction
        date_patterns = {
            "execution": r'(?:executed|signed|entered into).*?(?:on|as of)\s*([A-Za-z]+ \d{1,2},? \d{4})',
            "closing": r'(?:closing|completion).*?(?:on|by)\s*([A-Za-z]+ \d{1,2},? \d{4})'
        }
        
        for date_type, pattern in date_patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                date_text = match.group(1)
                if not any(date_text in cd["text"] for cd in color_dates):
                    dates_info[f"{date_type}_dates"].append({
                        "date": date_text,
                        "context": match.group(0),
                        "confidence": 0.8,
                        "source": "regex_pattern"
                    })
        
        return dates_info
    
    def _calculate_confidence_scores(self, legal_entities, financial_info, legal_references, has_colors) -> Dict[str, float]:
        """Calculate confidence scores (boosted if colors used)"""
        
        all_confidences = []
        
        # Collect all confidence scores
        for entity_type, entities in legal_entities.items():
            all_confidences.extend([e.get("confidence", 0) for e in entities])
        
        all_confidences.extend([a.get("confidence", 0) for a in financial_info.get("monetary_amounts", [])])
        
        for ref_type, refs in legal_references.items():
            all_confidences.extend([r.get("confidence", 0) for r in refs])
        
        overall = sum(all_confidences) / max(1, len(all_confidences))
        
        # Boost overall confidence if color markup was used
        if has_colors:
            overall = min(1.0, overall + 0.1)
        
        return {
            "overall_confidence": overall,
            "entity_confidence": sum([e.get("confidence", 0) for entities in legal_entities.values() for e in entities]) / max(1, sum(len(e) for e in legal_entities.values())),
            "financial_confidence": sum([a.get("confidence", 0) for a in financial_info.get("monetary_amounts", [])]) / max(1, len(financial_info.get("monetary_amounts", []))),
            "color_boost_applied": has_colors
        }
    
    def _deduplicate_entities(self, entities: Dict[str, List]) -> Dict[str, List]:
        """Remove duplicates, keep highest confidence"""
        for entity_type, entity_list in entities.items():
            seen = {}
            for entity in entity_list:
                text = entity["text"].lower().strip()
                if text not in seen or entity["confidence"] > seen[text]["confidence"]:
                    seen[text] = entity
            entities[entity_type] = list(seen.values())
        return entities
    
    def _create_annotated_document(self, original_doc: Document, annotations: Dict[str, Any]) -> Document:
        """Create annotated document with merged metadata"""
        enhanced_metadata = {
            **original_doc.metadata,
            "annotations": json.dumps(annotations, indent=2),
            "annotation_summary": {
                "total_entities": sum(len(e) for e in annotations["legal_entities"].values()),
                "total_amounts": len(annotations["financial_information"]["monetary_amounts"]),
                "confidence": annotations["confidence_scores"]["overall_confidence"],
                "color_integrated": annotations.get("color_integration_used", False)
            }
        }
        
        return Document(page_content=original_doc.page_content, metadata=enhanced_metadata)
    
    def _define_legal_patterns(self) -> Dict[str, str]:
        """Legal regex patterns"""
        return {
            "articles": r'ARTICLE\s+(?:[IVXLCDM]+|\d+)',
            "sections": r'(?:SECTION|Section)\s+\d+(?:\.\d+)*',
            "exhibits": r'(?:Exhibit|Schedule)\s+[A-Z\d]+'
        }
    
    def _define_financial_patterns(self) -> Dict[str, str]:
        """Financial regex patterns"""
        return {
            "dollar_amounts": r'\$[\d,]+(?:\.\d{2})?',
            "percentages": r'\d+(?:\.\d+)?%',
            "milestones": r'milestone.*?(?:\$[\d,]+|payment)'
        }


def annotate_documents(documents: List[Document]) -> List[Document]:
    """Main function called from rag_main.py"""
    engine = LegalAnnotationEngine()
    return engine.annotate_documents(documents)
