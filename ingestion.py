from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import os
import fitz
import pdfplumber
from collections import defaultdict
import re
from typing import Tuple


def ingestion_Documents(folder_path):
    """Load PDFs with color metadata extraction"""
    print(f"🎨 Color-aware loading: {folder_path}")
    documents = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            try:
                docs = extract_spa_document(file_path)
                documents.extend(docs)
                print(f"✅ {file_name} - {len(docs)} pages")
            except Exception as e:
                print(f"❌ {file_name}: {e}")
    
    print(f"✅ Total: {len(documents)} pages")
    return documents


def extract_spa_document(file_path):
    """Try PyMuPDF (with colors) → PDFplumber → PyPDF"""
    pymupdf_docs, pymupdf_score = try_pymupdf_with_colors(file_path)
    pdfplumber_docs, pdfplumber_score = try_pdfplumber(file_path)
    pypdf_docs, pypdf_score = try_pypdf(file_path)
    
    best_method, best_docs, best_score = max([
        ("PyMuPDF+Colors", pymupdf_docs, pymupdf_score),
        ("PDFplumber", pdfplumber_docs, pdfplumber_score),
        ("PyPDF", pypdf_docs, pypdf_score)
    ], key=lambda x: x[2])
    
    print(f"  📊 {best_method} (score: {best_score})")
    return best_docs if best_score > 0 else []


def try_pymupdf_with_colors(file_path):
    """PyMuPDF extraction with color metadata"""
    try:
        documents = []
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                color_data = extract_colors_and_annotations(page)
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_num + 1,
                        "extraction_method": "pymupdf_color",
                        "color_entities": color_data["entities"],
                        "color_categories": color_data["categories"],
                        "entity_counts": color_data["counts"]
                    }
                ))
        
        doc.close()
        return documents, score_with_colors(documents)
    except Exception as e:
        print(f"  ⚠️ PyMuPDF failed: {e}")
        return [], 0


def extract_colors_and_annotations(page):
    """Extract color metadata + highlighted annotations"""
    entities = []
    categories = defaultdict(list)
    counts = defaultdict(int)
    
    try:
        # Extract text colors
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    
                    # Convert color to RGB
                    c = span.get("color", 0)
                    rgb = (((c >> 16) & 0xFF) / 255, ((c >> 8) & 0xFF) / 255, (c & 0xFF) / 255)
                    
                    category = classify_color(rgb, text)
                    if category != "UNKNOWN":
                        entities.append({"text": text, "category": category, "rgb": rgb})
                        categories[category].append(text)
                        counts[category] += 1
        
        # Extract highlighted annotations
        annot = page.first_annot
        while annot:
            if annot.type[0] == 8:  # Highlight
                colors = annot.colors
                color_rgb = colors.get("stroke", colors.get("fill", None))
                if color_rgb:
                    highlighted_text = page.get_textbox(annot.rect).strip()
                    if highlighted_text:
                        category = classify_color(color_rgb, highlighted_text)
                        entities.append({"text": highlighted_text, "category": category, "type": "highlight"})
                        categories[category].append(highlighted_text)
                        counts[category] += 1
            annot = annot.next
    except Exception as e:
        print(f"    ⚠️ Color extraction error: {e}")
    
    return {"entities": entities, "categories": dict(categories), "counts": dict(counts)}


def classify_color(rgb: Tuple[float, float, float], text: str) -> str:
    """Map RGB + text context to SPA color category"""
    r, g, b = rgb
    text_lower = text.lower()
    
    # Color-based classification
    if r > 0.85 and g > 0.85 and b < 0.6:
        return "AMOUNT"  # Yellow
    if r < 0.6 and g > 0.75 and b < 0.6:
        return "PERCENT"  # Green
    if 0.65 < r < 0.85 and 0.65 < g < 0.85 and 0.65 < b < 0.85:
        return "DATE"  # Light Gray
    if 0.7 < r < 0.9 and g > 0.85 and 0.7 < b < 0.9:
        return "DURATION"  # Light Green
    if r > 0.85 and 0.65 < g < 0.9 and b > 0.75:
        return "DEFINED_TERM"  # Pink
    if 0.45 < r < 0.7 and g < 0.5 and b < 0.4:
        return "CROSSREF"  # Brown
    if r < 0.6 and g < 0.6 and b > 0.75:
        return "PARTY"  # Blue
    if 0.55 < r < 0.9 and 0.3 < g < 0.65 and 0.55 < b < 0.9:
        return "QUALIFIER"  # Purple
    
    # Text-based fallback
    if "$" in text or any(k in text_lower for k in ["dollar", "payment", "price"]):
        return "AMOUNT"
    if "%" in text:
        return "PERCENT"
    if any(k in text_lower for k in ["buyer", "seller", "purchaser"]):
        return "PARTY"
    
    return "UNKNOWN"


def try_pdfplumber(file_path):
    """PDFplumber extraction (no color)"""
    try:
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_num + 1, "extraction_method": "pdfplumber"}
                    ))
        return documents, score_extraction(documents)
    except:
        return [], 0


def try_pypdf(file_path):
    """PyPDF extraction (fallback)"""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents, score_extraction(documents)
    except:
        return [], 0


def score_extraction(docs):
    """Score SPA extraction quality"""
    if not docs:
        return 0
    
    text = ' '.join([d.page_content for d in docs]).lower()
    
    spa_kw = ['purchase agreement', 'consideration', 'seller', 'closing', 'shares', 'representations']
    fin_kw = ['$', 'cash', 'million', 'thousand']
    
    return (
        min(len(text) // 10, 10000) +
        sum(200 for k in spa_kw if k in text) +
        sum(500 for k in fin_kw if k in text) +
        len(re.findall(r'\$[\d,]+', text)) * 1000 +
        len(re.findall(r'\b[\d,]{6,}\b', text)) * 300
    )


def score_with_colors(docs):
    """Enhanced scoring with color bonus"""
    if not docs:
        return 0
    
    base = score_extraction(docs)
    color_bonus = 0
    
    for doc in docs:
        counts = doc.metadata.get("entity_counts", {})
        for cat, count in counts.items():
            bonus = 200 if cat in ["AMOUNT", "PERCENT"] else 150 if cat in ["PARTY", "DATE"] else 100
            color_bonus += count * bonus
    
    return base + color_bonus
