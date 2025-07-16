# Performance Testing Guide

## 🆕 New Features

### 📊 Graph Generation
- **All tests now generate graphs** - Quick, Vector, and Comprehensive tests
- **Performance dashboards** - Visual comparison of all model combinations
- **Cost analysis charts** - Cost vs performance trade-offs (comprehensive test)
- **Detailed comparison charts** - Heatmaps and distribution plots (comprehensive test)
- **Automatic graph saving** - All graphs saved as high-resolution PNG files

### 💾 Enhanced Output
- **Terminal summary saved** - All console output saved to text files for all tests
- **File location tracking** - Clear indication of where all results are saved
- **Automatic cleanup** - Old result files cleaned up before new runs (optional)

## Overview

This comprehensive performance testing framework helps you optimize your RAG system by comparing different algorithms, models, and configurations. **All tests now generate visual graphs** for easier analysis and better insights.

## 📊 Test Types

### 🏃 Quick Test (`quick_performance_test.py`) **✨ NEW: Now with Graphs!**

**Purpose:** Fast comparison of 3 key configurations
**Time:** 2-3 minutes
**Cost:** ~$0.10

**What it tests:**
- **Fast & Cheap:** `text-embedding-3-small` + `gpt-3.5-turbo` + 3 chunks
- **Balanced:** `text-embedding-3-large` + `gpt-4o-mini` + 5 chunks  
- **High Quality:** `text-embedding-3-large` + `gpt-4o` + 7 chunks

**🆕 Enhanced Output:**
- **Performance Dashboard** (`quick_test_*_dashboard.png`) - 4-panel analysis:
  - Total response time comparison
  - Time breakdown by component (setup, retrieval, generation)
  - Cost comparison per 1000 queries
  - Performance vs cost trade-off scatter plot
- **Terminal Summary** (`quick_test_*_terminal_summary.txt`) - All console output saved
- **JSON Results** (`quick_test_results_*.json`) - Raw data

**Use when:** You want a quick overview with visual analysis

### 🔍 Vector Benchmark (`vector_benchmark.py`) **✨ NEW: Now with Graphs!**

**Purpose:** Compare different FAISS vector search algorithms
**Time:** 5-10 minutes
**Cost:** ~$0.20

**What it tests:**
- **Flat L2:** Exact search with L2 distance
- **Flat IP:** Exact search with cosine similarity
- **IVF Flat:** Approximate search with clustering
- **IVF PQ:** Memory-efficient with product quantization
- **HNSW:** Hierarchical navigable small world graphs

**🆕 Enhanced Output:**
- **Performance Dashboard** (`vector_benchmark_*_dashboard.png`) - 4-panel analysis:
  - Average search time by index type
  - Index creation time comparison
  - Memory usage comparison
  - Performance vs memory trade-off scatter plot
- **Terminal Summary** (`vector_benchmark_*_terminal_summary.txt`) - All console output saved
- **JSON Results** (`vector_benchmark_*.json`) - Raw data

**Use when:** You want to optimize vector search with visual analysis

### 🧪 Comprehensive Test (`performance_tester.py`) **✨ ENHANCED FEATURES**

**Purpose:** Full matrix testing with advanced visual analysis
**Time:** 30-60 minutes
**Cost:** ~$2-5

**What it tests:**
- **3 Embedding models:** small, large, ada-002
- **3 LLM models:** gpt-3.5-turbo, gpt-4o-mini, gpt-4o
- **3 Search algorithms:** faiss_l2, faiss_cosine, basic_cosine
- **4 Chunk sizes:** 3, 5, 7, 10 chunks per query

**Total combinations:** 108 configurations tested

**🆕 Enhanced Output:**
- **Performance Dashboard** (`*_dashboard.png`) - 6-panel comprehensive analysis
- **Comparison Charts** (`*_comparison.png`) - Heatmaps and distribution plots
- **Cost Analysis** (`*_cost_analysis.png`) - Cost vs performance trade-offs
- **Terminal Summary** (`*_terminal_summary.txt`) - All console output saved
- **Markdown Report** (`*_summary.md`) - Detailed analysis with insights
- **CSV Data** (`*_results.csv`) - Raw data for further analysis
- **JSON Files** (`*_results.json`, `*_analysis.json`) - Structured data

**Use when:** You need detailed optimization data with advanced visual analysis

## 📈 Understanding Results

### 📊 Graph Types by Test

#### Quick Test Dashboard (`quick_test_*_dashboard.png`)
A 4-panel visualization showing:
1. **Total Response Time** - Bar chart comparing the 3 configurations
2. **Time Breakdown** - Stacked bar chart showing setup, retrieval, and generation times
3. **Cost Comparison** - Bar chart of estimated costs per 1000 queries
4. **Performance vs Cost** - Scatter plot showing the trade-off

#### Vector Benchmark Dashboard (`vector_benchmark_*_dashboard.png`)
A 4-panel visualization showing:
1. **Search Speed** - Bar chart comparing average search times
2. **Index Creation Time** - Bar chart showing creation speed
3. **Memory Usage** - Bar chart comparing memory requirements
4. **Performance vs Memory** - Scatter plot showing the trade-off

#### Comprehensive Test Dashboard (`*_dashboard.png`)
A comprehensive 6-panel visualization showing:
1. **Performance by Embedding Model** - Bar chart comparing average response times
2. **Performance by LLM Model** - Bar chart showing LLM speed differences
3. **Performance by Search Algorithm** - Bar chart of search algorithm performance
4. **Performance by Chunk Size** - Line chart showing optimal chunk size
5. **Time Breakdown** - Pie chart of retrieval vs generation time
6. **Top 10 Fastest Configurations** - Horizontal bar chart of best performers

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have:
- ✅ Processed content in `output_all/` directory 
- ✅ OpenAI API key in `.env` file
- ✅ All dependencies installed: `pip install -r requirements.txt`
- ✅ **New:** matplotlib and seaborn for graph generation

### 2. Run Performance Tests (All Generate Graphs!)

```bash
# Show available tests
python run_performance_tests.py

# Quick test with graphs (2-3 minutes, ~$0.10)
python run_performance_tests.py quick

# Vector search benchmark with graphs (5-10 minutes, ~$0.20)
python run_performance_tests.py vector

# Comprehensive test with advanced graphs (30-60 minutes, ~$2-5)
python run_performance_tests.py comprehensive
```

### 📋 File Output Structure

#### Quick Test Output:
```
quick_test_results_20240101_123456.json       # 📄 Raw results
quick_test_20240101_123456_dashboard.png      # 📊 Performance dashboard
quick_test_20240101_123456_terminal_summary.txt # 📋 Terminal output saved
```

#### Vector Benchmark Output:
```
vector_benchmark_20240101_123456.json         # 📄 Raw results
vector_benchmark_20240101_123456_dashboard.png # 📊 Performance dashboard
vector_benchmark_20240101_123456_terminal_summary.txt # 📋 Terminal output saved
```

#### Comprehensive Test Output:
```
performance_test_20240101_123456_dashboard.png        # 📊 Main performance dashboard
performance_test_20240101_123456_comparison.png       # 📊 Detailed comparison charts
performance_test_20240101_123456_cost_analysis.png    # 💰 Cost analysis charts
performance_test_20240101_123456_terminal_summary.txt # 📋 Terminal output saved
performance_test_20240101_123456_summary.md           # 📝 Detailed markdown report
performance_test_20240101_123456_results.csv          # 📊 Raw data for analysis
performance_test_20240101_123456_results.json         # 📄 Structured results
performance_test_20240101_123456_analysis.json        # 📊 Statistical analysis
```

## 🆕 Visual Analysis Workflow (All Tests)

### Quick Test Workflow:
1. **View Dashboard** (`quick_test_*_dashboard.png`)
   - Compare the 3 configurations visually
   - Check cost vs performance trade-off
   - Identify the fastest configuration

2. **Read Terminal Summary** (`quick_test_*_terminal_summary.txt`)
   - Get quick insights and recommendations
   - See exact timings and costs

### Vector Benchmark Workflow:
1. **View Dashboard** (`vector_benchmark_*_dashboard.png`)
   - Compare search speeds across algorithms
   - Check memory usage requirements
   - Find the best speed vs memory trade-off

2. **Read Terminal Summary** (`vector_benchmark_*_terminal_summary.txt`)
   - Get algorithm rankings and recommendations
   - See detailed performance metrics

### Comprehensive Test Workflow:
1. **Start with Dashboard** (`*_dashboard.png`)
   - Get overview of all model combinations
   - Identify best performers quickly

2. **Deep Dive with Comparison Charts** (`*_comparison.png`)
   - Analyze detailed heatmaps
   - Check performance distributions

3. **Optimize Costs** (`*_cost_analysis.png`)
   - Find budget-performance sweet spots
   - Compare model costs

4. **Read Reports**
   - **Terminal Summary** (`*_terminal_summary.txt`) - Quick reference
   - **Markdown Report** (`*_summary.md`) - Detailed insights

## 💡 Quick Tips

- **All tests generate graphs** - Open PNG files to visualize results
- **Terminal summaries saved** - Review TXT files for quick insights
- **Start with Quick Test** - Get visual overview in 2-3 minutes
- **Use Vector Benchmark** - Optimize search algorithms with graphs
- **Comprehensive for Production** - Get full analysis with advanced visualizations

Your RAG system performance testing now includes **visual analysis for all test types**! 🎨📊 

## 📊 Quick Reference: File Types

| File Type | Purpose | When to Use |
|-----------|---------|-------------|
| `*_dashboard.png` | Main visual overview | First look at results |
| `*_comparison.png` | Detailed comparisons | Deep analysis |
| `*_cost_analysis.png` | Cost optimization | Budget planning |
| `*_terminal_summary.txt` | Quick text summary | Fast reference |
| `*_summary.md` | Detailed report | Full analysis |
| `*_results.csv` | Raw data | Further analysis |
| `*_results.json` | Structured data | API integration |
| `*_analysis.json` | Statistical analysis | Data science |