# Performance Testing Guide

## Overview

This comprehensive performance testing framework helps you optimize your RAG system by comparing different algorithms, models, and configurations. It measures response times, costs, and quality across multiple dimensions.

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have:
- ✅ Processed content in `output_all/` directory 
- ✅ OpenAI API key in `.env` file
- ✅ All dependencies installed: `pip install -r requirements.txt`

### 2. Run Performance Tests

```bash
# Show available tests
python run_performance_tests.py

# Quick test (2-3 minutes, ~$0.10)
python run_performance_tests.py quick

# Vector search benchmark (5-10 minutes, ~$0.20)
python run_performance_tests.py vector

# Comprehensive test (30-60 minutes, ~$2-5)
python run_performance_tests.py comprehensive
```

## 📊 Test Types

### 🏃 Quick Test (`quick_performance_test.py`)

**Purpose:** Fast comparison of 3 key configurations
**Time:** 2-3 minutes
**Cost:** ~$0.10

**What it tests:**
- **Fast & Cheap:** `text-embedding-3-small` + `gpt-3.5-turbo` + 3 chunks
- **Balanced:** `text-embedding-3-large` + `gpt-4o-mini` + 5 chunks  
- **High Quality:** `text-embedding-3-large` + `gpt-4o` + 7 chunks

**Output:**
- Performance ranking by speed
- Cost estimates per 1000 queries
- Response quality preview
- JSON results file

**Use when:** You want a quick overview of performance trade-offs

### 🔍 Vector Benchmark (`vector_benchmark.py`)

**Purpose:** Compare different FAISS vector search algorithms
**Time:** 5-10 minutes
**Cost:** ~$0.20

**What it tests:**
- **Flat L2:** Exact search with L2 distance
- **Flat IP:** Exact search with cosine similarity
- **IVF Flat:** Approximate search with clustering
- **IVF PQ:** Memory-efficient with product quantization
- **HNSW:** Hierarchical navigable small world graphs

**Metrics:**
- Index creation time
- Search speed (milliseconds)
- Memory usage
- Training time (for applicable algorithms)

**Use when:** You want to optimize vector search performance specifically

### 🧪 Comprehensive Test (`performance_tester.py`)

**Purpose:** Full matrix testing of all configurations
**Time:** 30-60 minutes
**Cost:** ~$2-5

**What it tests:**
- **3 Embedding models:** small, large, ada-002
- **3 LLM models:** gpt-3.5-turbo, gpt-4o-mini, gpt-4o
- **3 Search algorithms:** faiss_l2, faiss_cosine, basic_cosine
- **4 Chunk sizes:** 3, 5, 7, 10 chunks per query

**Total combinations:** 108 configurations tested

**Output:**
- Detailed CSV with all results
- Statistical analysis by model type
- Best configurations for different criteria
- Comprehensive markdown report

**Use when:** You need detailed optimization data for production deployment

## 📈 Understanding Results

### Performance Metrics

#### Response Time Components
- **Retrieval Time:** How fast the system finds relevant chunks
- **Generation Time:** How fast the LLM generates the response
- **Total Time:** End-to-end response time
- **Setup Time:** Index creation time (one-time cost)

#### Quality Indicators
- **Chunks Retrieved:** Number of relevant documents found
- **Response Length:** Length of generated response
- **Relevance Scores:** Similarity scores of retrieved chunks

#### Cost Factors
- **Embedding Costs:** Cost per token for creating embeddings
- **LLM Costs:** Cost per token for generating responses
- **Total Cost:** Estimated cost per 1000 queries

### Sample Results Interpretation

```
🏆 BEST CONFIGURATION:
   text-embedding-3-large + faiss_l2 + gpt-4o-mini (k=5)
   Response time: 1.234s

💰 COST ANALYSIS:
   Fast & Cheap: ~$1.20 per 1000 queries
   Balanced: ~$3.50 per 1000 queries  
   High Quality: ~$15.00 per 1000 queries
```

**Interpretation:**
- **1.234s response time** is excellent for production use
- **$3.50 per 1000 queries** offers good cost/performance balance
- **k=5 chunks** provides optimal context without excess

## ⚙️ Advanced Usage

### Custom Test Configurations

#### Modify Test Queries
Edit the `test_queries` list in any test script:

```python
self.test_queries = [
    "Your custom query 1",
    "Your custom query 2", 
    "Your custom query 3"
]
```

#### Add New Models
Update model lists in `performance_tester.py`:

```python
self.embedding_models = [
    "text-embedding-3-small",
    "text-embedding-3-large", 
    "your-custom-embedding-model"
]

self.llm_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "your-custom-llm-model"
]
```

#### Custom Vector Algorithms
Add new algorithms in `vector_benchmark.py`:

```python
self.index_types = {
    "Your Algorithm": "your_algorithm_key"
}
```

### Running Specific Configurations

#### Test Only Fast Models
```bash
python performance_tester.py --sample-size 2 --output fast_models
```

#### Compare Embedding Models Only
Modify `performance_tester.py` to test only embedding variations:

```python
self.llm_models = ["gpt-4o-mini"]  # Fix LLM
self.search_algorithms = ["faiss_l2"]  # Fix algorithm  
self.chunk_sizes = [5]  # Fix chunk size
```

## 📊 Analyzing Results

### Generated Files

Each test creates multiple output files:

#### Quick Test
- `quick_test_results_TIMESTAMP.json` - Raw results
- Terminal output with rankings and recommendations

#### Vector Benchmark  
- `vector_benchmark_TIMESTAMP.json` - Detailed index performance
- Terminal analysis with rankings

#### Comprehensive Test
- `performance_test_TIMESTAMP_results.json` - Raw test data
- `performance_test_TIMESTAMP_analysis.json` - Statistical analysis
- `performance_test_TIMESTAMP_results.csv` - Spreadsheet format
- `performance_test_TIMESTAMP_summary.md` - Human-readable report

### Key Questions to Answer

#### 1. What's the fastest configuration?
Look for `fastest_overall` in analysis results or top-ranked by total time.

#### 2. What's the most cost-effective?
Compare cost estimates in quick test or calculate from token usage.

#### 3. What's the best for production?
Consider balance of speed, cost, and quality. Usually "Balanced" configuration wins.

#### 4. How much will scaling cost?
Multiply per-query costs by expected daily query volume.

### Performance Optimization Tips

#### For Speed Optimization:
1. Use `text-embedding-3-small` for embeddings
2. Use `gpt-4o-mini` for generation  
3. Reduce chunk count to 3-5
4. Use `faiss_l2` for search

#### For Cost Optimization:
1. Use `gpt-3.5-turbo` for generation
2. Use smaller embedding models
3. Optimize chunk size (not always smaller = cheaper)
4. Cache frequent queries

#### For Quality Optimization:
1. Use `text-embedding-3-large` for embeddings
2. Use `gpt-4o` for generation
3. Increase chunk count to 7-10
4. Experiment with cosine similarity search

#### For Memory Optimization:
1. Use IVF or PQ vector indexes
2. Smaller embedding dimensions
3. Product quantization for vector compression

## 🎯 Production Recommendations

### Small Scale (< 1000 queries/day)
- **Configuration:** Fast & Cheap
- **Expected cost:** ~$1-2/day
- **Response time:** ~2-3 seconds

### Medium Scale (1000-10,000 queries/day)  
- **Configuration:** Balanced
- **Expected cost:** ~$35-70/day
- **Response time:** ~1-2 seconds

### Large Scale (10,000+ queries/day)
- **Configuration:** Custom optimized based on comprehensive test
- **Expected cost:** Negotiate enterprise pricing
- **Response time:** ~0.5-1 seconds with optimizations

### Real-time Applications (< 500ms requirement)
- **Pre-computed embeddings:** Store in fast database
- **Streaming responses:** Implement LLM streaming
- **Caching:** Cache frequent query patterns
- **Load balancing:** Multiple model instances

## 🐛 Troubleshooting

### Common Issues

#### "No documents found"
```bash
# Check if you have processed content
ls output_all/*/chunks/

# If empty, process some content first:
python chunkr.py -d data -o output_all
# or
python webscraper.py -f sample_urls.txt -o output_all -n web_content
```

#### "OpenAI API key not found"
```bash
# Check .env file exists
cat .env

# Should contain:
OPENAI_API_KEY=your_key_here
```

#### "ModuleNotFoundError"
```bash
# Install all dependencies
pip install -r requirements.txt
```

#### Tests running slowly
- Reduce sample size: `--sample-size 1`
- Test fewer models by editing the script
- Use quick test instead of comprehensive

#### High API costs
- Start with quick test to get estimates
- Use smaller sample sizes for comprehensive tests
- Monitor OpenAI usage dashboard

### Performance Issues

#### Slow vector search
- Try different FAISS indexes (IVF for large datasets)
- Reduce embedding dimensions
- Consider approximate search algorithms

#### Slow LLM responses
- Use smaller models (gpt-4o-mini vs gpt-4o)
- Reduce context length (fewer chunks)
- Implement response streaming

#### Memory issues
- Use IVF or PQ vector indexes
- Process documents in smaller batches
- Use smaller embedding models

## 📚 Further Reading

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [LangChain Performance Tips](https://python.langchain.com/docs/guides/performance)
- [Vector Database Comparisons](https://benchmark.vectorview.ai/)

## 🤝 Contributing

To add new test types or algorithms:

1. Create a new test script following existing patterns
2. Add it to `run_performance_tests.py`
3. Update this guide with usage instructions
4. Test with different data sizes and configurations

Your RAG system performance testing is now comprehensive and production-ready! 🚀 