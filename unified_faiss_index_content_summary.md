# RAG System Content Summary

**Generated:** 2025-07-01T15:46:23.214833
**Index Name:** unified_faiss_index

## Content Sources

### Kansas Motorcycle Handbook
- **Chunks:** 72

### kansas_dmv_web
- **Chunks:** 7
- **Summary:** scraping_summary.md

### Kansas Driving Handbook
- **Chunks:** 218

### Kansas Commercial Driver's License Manual
- **Chunks:** 456

### kansas_web
- **Chunks:** 2
- **Summary:** scraping_summary.md

## Usage Instructions

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load the unified index
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = FAISS.load_local('unified_faiss_index', embeddings, allow_dangerous_deserialization=True)

# Search across all content
results = vector_store.similarity_search('your query here', k=5)
```
