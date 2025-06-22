# PDF Multi-Document Processing and Search System

A comprehensive system for processing multiple PDF documents, creating embeddings, and enabling semantic search across all documents using Chunkr AI and OpenAI.

## Features

- **Multi-PDF Processing**: Process multiple PDFs in a folder simultaneously
- **Intelligent Chunking**: Uses Chunkr AI for optimal text chunking with 1024-token targets
- **Cost-Effective Configuration**: Minimizes LLM usage by only processing tables with LLM, text uses fast heuristics
- **Unified Search**: Search across all processed documents with a single query
- **File-Specific Search**: Search within specific documents
- **Rich Metadata**: Track source files, chunk positions, segment types, and more
- **Interactive Mode**: Command-line interface for real-time searching
- **Export Capabilities**: Export processed content to markdown and text formats

## Prerequisites

- Python 3.8+
- Chunkr AI API key ([get here](https://chunkr.ai))
- OpenAI API key ([get here](https://platform.openai.com))

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
4. Edit `.env` and add your API keys:
   ```
   CHUNKR_API_KEY=your_chunkr_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   PDF_FOLDER=./data
   INDEX_NAME=unified_index
   ```

## Usage

### Basic Usage

1. Place your PDF files in the `./data` folder (or specify a different folder in `.env`)
2. Run the system:
   ```bash
   python Chunker.py
   ```

### What the System Does

1. **PDF Processing**: Finds all PDFs in the specified folder and processes them using Chunkr AI
2. **Embedding Creation**: Creates OpenAI embeddings for all text chunks
3. **FAISS Index**: Builds a unified search index for fast similarity search
4. **Demo Searches**: Runs example searches to demonstrate capabilities
5. **Interactive Mode**: Provides a command-line interface for custom searches

### Search Features

#### Cross-Document Search
```python
# Search across all documents
results = search_engine.search("regulatory requirements", k=5)
```

#### File-Specific Search
```python
# Search within a specific document
results = search_engine.search_by_file("methodology", "document.pdf", k=3)
```

#### Interactive Search
In interactive mode, you can:
- Type any query to search across all documents
- Use `file:filename:query` to search within a specific file
- Type `files` to list all available documents
- Type `quit` to exit

### Configuration Details

The system uses a cost-effective configuration:

- **Chunking**: 1024 tokens per chunk using OpenAI's tokenizer
- **LLM Usage**: Only tables are processed with LLM (GPT-4o-mini) for cost efficiency
- **Text Processing**: Headers, titles, and regular text use fast AUTO processing
- **Embeddings**: Uses `text-embedding-3-small` (1536 dimensions) for cost-effectiveness
- **Search**: FAISS with cosine similarity for fast retrieval

## Output Files

The system generates several output files:

- `{INDEX_NAME}.index` - FAISS search index
- `{INDEX_NAME}_metadata.pkl` - Chunk metadata (binary)
- `{INDEX_NAME}_metadata.json` - Chunk metadata (human-readable)
- `unified_documents.md` - Combined markdown from all PDFs
- `unified_documents_chunks.txt` - All chunks with source information

## Code Structure

### Main Components

1. **`get_pdf_config()`** - Configures Chunkr processing parameters
2. **`process_multiple_pdfs()`** - Processes all PDFs in a folder
3. **`create_unified_faiss_index()`** - Creates embeddings and FAISS index
4. **`UnifiedFAISSSearch`** - Main search class with methods:
   - `search()` - Cross-document search
   - `search_by_file()` - File-specific search
   - `get_file_stats()` - Document statistics
   - `list_files()` - Available documents
5. **`demo_unified_search()`** - Demonstration searches
6. **`interactive_search()`** - Interactive command-line interface
7. **`main()`** - Main orchestration function

### Example Usage in Code

```python
from Chunker import UnifiedFAISSSearch

# Load existing index
search_engine = UnifiedFAISSSearch("unified_index", openai_api_key)

# Search across all documents
results = search_engine.search("machine learning algorithms", k=5)

# Search within specific document
results = search_engine.search_by_file("neural networks", "research_paper.pdf", k=3)

# Get document statistics
stats = search_engine.get_file_stats()
```

## Performance Characteristics

- **Processing Speed**: ~1-2 seconds per PDF page (varies by content)
- **Search Speed**: ~100-500ms per query (after index is loaded)
- **Memory Usage**: ~100MB for 1000 chunks + index size
- **Storage**: ~1-5MB per 100 chunks (index + metadata)

## Cost Estimates

- **Chunkr Processing**: ~$0.001-0.01 per page (depending on table content)
- **OpenAI Embeddings**: ~$0.0001 per 1000 tokens
- **Total**: Typically $0.01-0.10 per document

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file contains valid API keys
2. **No PDFs Found**: Check that PDFs are in the correct folder
3. **Memory Issues**: For large document sets, process in smaller batches
4. **Search Returns No Results**: Try broader or different search terms

### Error Handling

The system includes comprehensive error handling:
- Failed PDF processing doesn't stop the entire batch
- Network errors are caught and reported
- Invalid search queries are handled gracefully

## Dependencies

- `chunkr-ai`: PDF processing and chunking
- `openai`: Embeddings and API access
- `numpy`: Numerical operations for embeddings
- `faiss-cpu`: Fast similarity search
- `python-dotenv`: Environment variable management

## License

This project is provided as-is for educational and research purposes.
