# Color-Aware RAG Pipeline for Legal Documents

A sophisticated **Retrieval-Augmented Generation (RAG)** system designed specifically for analyzing legal documents, particularly **Stock Purchase Agreements (SPAs)**. This system uniquely leverages **color-coded metadata** extracted from PDF documents to enhance entity recognition, improve retrieval accuracy, and generate more precise legal analysis.

## 🎯 Overview

This pipeline processes legal PDF documents through a multi-stage pipeline that:
1. **Extracts color-coded annotations** from PDFs (highlighted text, colored entities)
2. **Transforms documents** into structured JSON with metadata
3. **Annotates entities** using both color metadata and NLP (spaCy)
4. **Chunks documents** while preserving color and annotation data
5. **Creates vector embeddings** and stores them in Weaviate
6. **Retrieves and generates** answers using color-aware filtering

### Key Innovation: Color-Aware Processing

Unlike traditional RAG systems that rely solely on text, this system extracts and utilizes **color-coded metadata** from PDFs:
- **Yellow** = Dollar amounts (AMOUNT)
- **Green** = Percentages (PERCENT)
- **Blue** = Party names (PARTY)
- **Gray** = Dates (DATE)
- **Pink** = Defined terms (DEFINED_TERM)
- **Brown** = Cross-references (CROSSREF)
- **Purple** = Qualifiers (QUALIFIER)

This color metadata provides **high-confidence entity identification** (0.95 confidence) compared to regex-based extraction (0.75-0.8 confidence).

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    COLOR-AWARE RAG PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

Stage 1: DOCUMENT INGESTION (ingestion.py)
├── Load PDFs from folder
├── Extract text using PyMuPDF (with color extraction)
├── Fallback to PDFplumber or PyPDF if needed
├── Extract color metadata:
│   ├── Text colors (RGB values)
│   ├── Highlighted annotations
│   └── Classify colors into categories
└── Output: Documents with color_entities, color_categories metadata

Stage 2: JSON TRANSFORMATION (json_transformer.py)
├── Extract structured metadata:
│   ├── Document title, type, parties
│   ├── Purchase prices (prioritizing color-coded AMOUNT)
│   ├── Dates (prioritizing color-coded DATE)
│   └── Legal sections, companies
├── Priority: Color metadata (0.95 confidence) > Regex (0.75 confidence)
└── Output: Documents with structured_data JSON

Stage 3: LEGAL ANNOTATION (annotation_engine.py)
├── Combine color metadata + spaCy NER + Regex patterns
├── Extract:
│   ├── Legal entities (companies, persons, roles)
│   ├── Financial information (amounts, percentages, payment structures)
│   ├── Legal references (articles, sections, exhibits)
│   └── Dates and deadlines
├── Calculate confidence scores (boosted if color metadata used)
└── Output: Documents with annotations JSON

Stage 4: TEXT CHUNKING (splitter.py)
├── RecursiveCharacterTextSplitter (4000 chars, 800 overlap)
├── Preserve color metadata per chunk
├── Add chunk-level metadata:
│   ├── has_color_amounts, has_color_parties, has_color_dates
│   ├── color_entity_count
│   ├── annotation_confidence
│   └── relevance_score
└── Output: Enhanced chunks with color/annotation flags

Stage 5: EMBEDDING & VECTOR STORAGE (embedding_vector_doc.py)
├── Generate embeddings using OpenAI text-embedding-3-large
├── Connect to Weaviate vector database
├── Store chunks with all metadata (color, annotations, structured data)
├── Batch processing (25 chunks per batch)
└── Output: WeaviateVectorStore ready for querying

Stage 6: RETRIEVAL & GENERATION (retrieval_and_generation.py)
├── Recursive retrieval (3-5 iterations for complex queries)
├── Color-aware metadata filtering:
│   ├── has_color_amounts=true for financial queries
│   ├── has_color_parties=true for party queries
│   └── has_color_dates=true for date queries
├── Enhanced context formatting (includes color/annotation metadata)
├── LLM generation with color-aware prompt
└── Output: RAG chain ready for querying
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Weaviate)
- OpenAI API key
- spaCy model (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd coloured_doc_code
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy model** (optional, for enhanced NER)
   ```bash
   python -m spacy download en_core_web_md
   # or
   python -m spacy download en_core_web_lg
   ```

4. **Set up environment variables**
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
   WEAVIATE_URL=http://localhost:8081
   WEAVIATE_INDEX_NAME=LegalDocuments
   ```

5. **Start Weaviate vector database**
   ```bash
   docker-compose up -d
   ```

6. **Prepare your documents**
   Place PDF files in a folder (e.g., `40_SPAs/`)

---

## 📖 Usage

### Full Pipeline Execution

Run the complete pipeline from ingestion to RAG setup:

```bash
python rag_main.py
```

This will:
1. Load all PDFs from `40_SPAs/` folder
2. Extract color metadata
3. Transform to structured JSON
4. Annotate with legal entities
5. Chunk documents
6. Create embeddings and store in Weaviate
7. Set up the RAG pipeline

### Query the RAG System

#### Single Query Testing
```bash
python query_rag.py
```

#### Batch Query Testing
```bash
python batch_query_test.py
# or with text output
python batch_query_test.py txt
```

The batch test reads queries from `all_queries.txt` and generates results in JSON or text format.

#### Test Single Document
```bash
python test_single_colour_doc.py "path/to/document.pdf"
```

### Example Queries

```python
from query_rag import connect_to_vectorstore, retrieve_and_generate

# Connect to vectorstore
vectorstore, client = connect_to_vectorstore()
rag_pipeline = retrieve_and_generate(vectorstore)

# Query examples
queries = [
    "What is the purchase price for Heat Biologics shares?",
    "Who are the buyer and seller in the Ariba Inc. acquisition?",
    "What is the cash consideration in the Q2 Software transaction?",
    "Describe the indemnification cap in the Pilgrim's Pride agreement."
]

for query in queries:
    answer = rag_pipeline.invoke(query)
    print(f"Q: {query}\nA: {answer}\n")
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `OPENAI_MODEL` | LLM model for generation | `gpt-4` |
| `WEAVIATE_URL` | Weaviate server URL | `http://localhost:8081` |
| `WEAVIATE_INDEX_NAME` | Vector store index name | `LegalDocuments` |

### Chunking Parameters

Edit `splitter.py` to adjust:
- `chunk_size`: 4000 characters (default)
- `chunk_overlap`: 800 characters (default)
- Separators: Article/Section boundaries prioritized

### Retrieval Parameters

Edit `retrieval_and_generation.py` to adjust:
- `k_value`: Number of documents retrieved (25-45 based on iteration)
- `max_iterations`: Recursive retrieval depth (3-5 based on query complexity)
- `context_limit`: Maximum contexts used (25-35)

---

## 📁 Project Structure

```
coloured_doc_code/
├── rag_main.py                 # Main orchestration script
├── ingestion.py                # PDF loading with color extraction
├── json_transformer.py         # Structured JSON transformation
├── annotation_engine.py        # Legal entity annotation (color + NLP)
├── splitter.py                 # Text chunking with metadata preservation
├── embedding_vector_doc.py     # Embedding generation & Weaviate storage
├── retrieval_and_generation.py # RAG chain with color-aware retrieval
├── query_rag.py               # Single query interface
├── batch_query_test.py        # Batch testing script
├── test_single_colour_doc.py  # Single document testing
├── docker-compose.yml         # Weaviate configuration
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🎨 Color Metadata System

### Color Categories

The system recognizes and classifies the following color-coded entities:

| Color | Category | Description | Confidence |
|-------|----------|-------------|------------|
| 🟡 Yellow | `AMOUNT` | Dollar amounts, purchase prices | 0.95 |
| 🟢 Green | `PERCENT` | Percentages, ratios | 0.95 |
| 🔵 Blue | `PARTY` | Company names, buyer/seller | 0.95 |
| ⚪ Gray | `DATE` | Execution dates, closing dates | 0.95 |
| 🩷 Pink | `DEFINED_TERM` | Legal definitions | 0.95 |
| 🤎 Brown | `CROSSREF` | Article/section references | 0.95 |
| 🟣 Purple | `QUALIFIER` | Material, knowledge qualifiers | 0.95 |

### Color Extraction Process

1. **Text Color Extraction**: Uses PyMuPDF to extract RGB values from text spans
2. **Highlight Extraction**: Extracts highlighted annotations with their colors
3. **Classification**: Maps RGB values to semantic categories
4. **Context Analysis**: Uses surrounding text to disambiguate categories
5. **Confidence Scoring**: Color-marked entities receive 0.95 confidence vs 0.75 for regex

### Benefits

- **Higher Accuracy**: Color-coded entities are human-verified, reducing false positives
- **Better Retrieval**: Metadata filters like `has_color_amounts=true` improve precision
- **Enhanced Context**: LLM receives explicit entity type information
- **Confidence Tracking**: System knows which entities are high-confidence

---

## 🔍 Retrieval Strategy

### Recursive Retrieval

The system uses an adaptive recursive retrieval strategy:

1. **Initial Query**: Broad search with color-aware metadata filters
2. **Context Analysis**: LLM analyzes retrieved context for gaps
3. **Follow-up Queries**: Generate targeted queries for missing information
4. **Iterative Retrieval**: Repeat 3-5 times for complex queries
5. **Deduplication**: Remove duplicate contexts while preserving metadata

### Metadata Filtering

The system automatically applies filters based on query type:

```python
# Financial queries → Filter for color-coded amounts
{"has_color_amounts": true, "contains_financial_info": true}

# Party queries → Filter for color-coded parties
{"has_color_parties": true}

# Date queries → Filter for color-coded dates
{"has_color_dates": true}
```

### Enhanced Context Formatting

Retrieved documents include:
- Structured metadata (title, type, parties, purchase price)
- Annotation data (entities, amounts, confidence scores)
- Color metadata (entity categories, counts)
- Document hierarchy (articles, sections)

---

## 🧪 Testing

### Single Document Test

Test the pipeline with a single PDF:

```bash
python test_single_colour_doc.py "40_SPAs/31.RECTORSEAL, LLC (1).pdf"
```

This will:
- Run the full pipeline on one document
- Display color metadata statistics
- Run sample queries
- Show annotation summaries

### Batch Testing

Run all queries from `all_queries.txt`:

```bash
python batch_query_test.py json  # JSON output
python batch_query_test.py txt   # Text output
```

Results are saved with timestamps:
- `batch_test_results_YYYYMMDD_HHMMSS.json`
- `batch_test_results_YYYYMMDD_HHMMSS.txt`

---

## 📊 Performance Features

### Confidence Scoring

The system tracks confidence at multiple levels:

- **Entity Confidence**: 0.95 for color-marked, 0.75 for regex
- **Annotation Confidence**: Overall document annotation quality
- **Chunk Relevance**: Calculated from color entities + annotations + content

### Quality Indicators

Chunks are marked with quality flags:
- `high_quality_chunk`: Overall confidence > 0.8
- `has_annotations`: Contains legal entity annotations
- `has_color_amounts`: Contains color-coded financial data
- `relevance_score`: 0.0-1.0 based on multiple factors

### Statistics Tracking

The pipeline tracks:
- Color entity counts by category
- Annotation coverage
- Financial data presence
- Chunk quality distribution

---

## 🔄 Integration Points

### Weaviate Vector Database

- **Index**: `LegalDocuments` (configurable)
- **Embeddings**: OpenAI `text-embedding-3-large`
- **Metadata**: All color, annotation, and structured data preserved
- **Filtering**: Supports complex metadata filters for retrieval

### OpenAI Integration

- **Embeddings**: `text-embedding-3-large` (3072 dimensions)
- **LLM**: Configurable (default: `gpt-4`)
- **Temperature**: 0.01 for consistent legal analysis

### spaCy Integration

- **Model**: `en_core_web_md` or `en_core_web_lg`
- **Purpose**: Named Entity Recognition (ORG, PERSON)
- **Fallback**: Works without spaCy (color + regex only)

---

## 🛠️ Troubleshooting

### Common Issues

1. **Weaviate Connection Error**
   - Ensure Docker is running: `docker-compose up -d`
   - Check port 8081 is available
   - Verify `WEAVIATE_URL` in `.env`

2. **OpenAI API Errors**
   - Verify `OPENAI_API_KEY` in `.env`
   - Check API quota/rate limits
   - Ensure model name is correct

3. **Color Extraction Not Working**
   - PDFs must have actual color annotations (not just text)
   - Try different PDF extraction method (PyMuPDF → PDFplumber → PyPDF)
   - Check if PDFs are scanned images (OCR may be needed)

4. **spaCy Model Not Found**
   - Run: `python -m spacy download en_core_web_md`
   - System will fallback to regex if spaCy unavailable

5. **Memory Issues with Large Documents**
   - Reduce `chunk_size` in `splitter.py`
   - Process documents in smaller batches
   - Increase Docker memory allocation

---

## 📈 Future Enhancements

Potential improvements:

- [ ] Support for additional document types (merger agreements, asset purchase agreements)
- [ ] Fine-tuned embeddings for legal domain
- [ ] Multi-modal extraction (tables, images)
- [ ] Advanced cross-document relationship mapping
- [ ] Real-time document ingestion API
- [ ] Web interface for querying
- [ ] Export capabilities (JSON, CSV, Excel)
- [ ] Confidence-based answer ranking

---

## 📝 License

[Specify your license here]

---

## 🤝 Contributing

[Contributing guidelines if applicable]

---

## 📧 Contact

[Contact information if applicable]

---

## 🙏 Acknowledgments

- **LangChain**: RAG pipeline framework
- **Weaviate**: Vector database
- **OpenAI**: Embeddings and LLM
- **PyMuPDF**: PDF and color extraction
- **spaCy**: Named Entity Recognition

---

**Built with ❤️ for legal document analysis**
