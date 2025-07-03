# Web Scraper Integration Guide

## Overview

The web scraper (`webscraper.py`) extends your RAG system to include content from websites alongside PDF documents. This allows you to create a more comprehensive knowledge base by combining official documents with online resources.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install requests beautifulsoup4 lxml
```

### 2. Basic Usage

#### Scrape specific URLs:
```bash
python webscraper.py -u https://www.ksrevenue.gov/dmv.html https://www.dmv.org/ks-kansas/ -o output_all -n kansas_web
```

#### Scrape URLs from file:
```bash
python webscraper.py -f sample_urls.txt -o output_all -n kansas_dmv_web
```

### 3. Combine with existing content:
```bash
python embed_and_store_combined.py
```

### 4. Run your RAG system:
```bash
python gradio_app.py
```

## 📖 Detailed Usage

### Command Line Options

```bash
python webscraper.py [OPTIONS]
```

**Input Options (choose one):**
- `-u, --urls URL1 URL2 ...` - List of URLs to scrape
- `-f, --file FILE` - Text file containing URLs (one per line)

**Output Options:**
- `-o, --output DIR` - Output directory (required)
- `-n, --name NAME` - Name for content source (default: "web_content")

**Scraping Options:**
- `-d, --delay SECONDS` - Delay between requests (default: 1.0)
- `-r, --retries NUM` - Maximum retries for failed requests (default: 3)

### Examples

#### Example 1: Scrape Kansas DMV Resources
```bash
python webscraper.py \
  -u https://www.ksrevenue.gov/dmv.html \
     https://www.ksrevenue.gov/dmvdl.html \
     https://www.ksrevenue.gov/dmvvr.html \
  -o output_all \
  -n kansas_official \
  -d 2.0
```

#### Example 2: Scrape from URL file
```bash
# Create urls.txt with your URLs
echo "https://www.nhtsa.gov/road-safety" > urls.txt
echo "https://www.dmv.org/ks-kansas/" >> urls.txt

python webscraper.py -f urls.txt -o output_all -n safety_resources
```

## 📁 Output Structure

The scraper creates the same structure as PDF processing:

```
output_all/
├── kansas_dmv_web/
│   ├── chunks/
│   │   ├── chunk_0.md
│   │   ├── chunk_1.md
│   │   └── ...
│   ├── chunks.json
│   ├── original_scraped_data.json
│   └── scraping_summary.md
└── [other_content_sources]/
```

## 🔄 Integration Workflow

### Complete Setup Process:

1. **Process PDFs** (if you have them):
   ```bash
   python chunkr.py -d data -o output_all
   ```

2. **Scrape Websites**:
   ```bash
   python webscraper.py -f sample_urls.txt -o output_all -n web_content
   ```

3. **Create Unified Index**:
   ```bash
   python embed_and_store_combined.py
   ```

4. **Update Agent** (if needed):
   ```python
   # In agent.py, change:
   # faiss_index_path: str = "faiss_index"
   # to:
   # faiss_index_path: str = "unified_faiss_index"
   ```

5. **Launch Web Interface**:
   ```bash
   python gradio_app.py
   ```

## 🛡️ Ethical Scraping Features

The scraper includes several ethical safeguards:

- **Robots.txt Compliance**: Checks and respects robots.txt files
- **Rate Limiting**: Configurable delays between requests
- **Respectful User Agent**: Identifies itself as educational/research tool
- **Retry Logic**: Exponential backoff for failed requests
- **Duplicate Prevention**: Avoids scraping the same URL twice

## 📝 URL File Format

Create a text file with URLs (one per line):

```
# Comments start with #
https://www.ksrevenue.gov/dmv.html
https://www.ksrevenue.gov/dmvdl.html

# You can group URLs by topic
# Motor Vehicle Safety
https://www.nhtsa.gov/road-safety
https://www.dmv.org/ks-kansas/

# Commercial Driving
https://www.fmcsa.dot.gov/registration/commercial-drivers-license
```

## 🔍 Content Processing

### What Gets Extracted:
- **Page Title**: From `<title>` tag
- **Main Content**: Prioritizes `<main>`, `<article>`, content areas
- **Meta Description**: For additional context
- **Clean Text**: Removes navigation, headers, footers, scripts

### What Gets Filtered Out:
- Navigation menus
- Headers and footers
- Sidebar content
- Scripts and styles
- Very short text segments

### Chunking Strategy:
- **Chunk Size**: ~1000 words per chunk (matching PDF processing)
- **Metadata**: Includes source URL, domain, scraping timestamp
- **Format**: Markdown with metadata headers

## 🧪 Testing Your Setup

Test individual components:

1. **Test Web Scraper**:
   ```bash
   python webscraper.py -u https://www.ksrevenue.gov/dmv.html -o test_output -n test
   ```

# Scrape specific URLs
python webscraper.py -u https://www.ksrevenue.gov/dmv.html https://www.dmv.org/ks-kansas/ -o output_all -n kansas_web

# Scrape from URL file
python webscraper.py -f sample_urls.txt -o output_all -n kansas_dmv_web

# Create unified index
python embed_and_store_combined.py

# Custom rate limiting and retry settings
python webscraper.py -f urls.txt -o output_all -n web_content -d 2.0 -r 5

2. **Test Combined Index**:
   ```bash
   python embed_and_store_combined.py
   ```

3. **Test Search**:
   ```python
   from langchain_community.vectorstores import FAISS
   from langchain_openai import OpenAIEmbeddings
   
   embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
   vector_store = FAISS.load_local("unified_faiss_index", embeddings, allow_dangerous_deserialization=True)
   
   results = vector_store.similarity_search("Kansas driving license", k=3)
   for result in results:
       print(f"Source: {result.metadata.get('source_type', 'unknown')}")
       print(f"Content: {result.page_content[:200]}...")
   ```

## 🐛 Troubleshooting

### Common Issues:

1. **No content extracted**:
   - Check if the website blocks scrapers
   - Verify URLs are accessible
   - Some sites may require JavaScript (not supported)

2. **Robots.txt blocked**:
   - Respect the website's robots.txt
   - Look for alternative sources
   - Contact site owner for permission

3. **Empty chunks**:
   - Some pages may have minimal text content
   - Adjust content filtering in the scraper if needed

4. **Rate limiting**:
   - Increase delay between requests (`-d` option)
   - Some sites may temporarily block rapid requests

### Debug Mode:
Add print statements in `webscraper.py` to see what content is being extracted.

## 🔧 Customization

### Adding New Content Selectors:
In `webscraper.py`, modify the `content_selectors` list:

```python
content_selectors = [
    'main', 'article', '.content', '#content',
    '.main-content', '#main-content', '.post-content',
    '.entry-content', '.article-content',
    # Add your custom selectors here
    '.your-custom-class', '#your-custom-id'
]
```

### Adjusting Chunk Size:
In `save_scraped_content()` function, modify:

```python
chunk_size = 1000  # Change this value
```

### Custom Filtering:
Modify the `clean_text()` function to add custom text cleaning rules.

## 📊 Performance Tips

1. **Batch Processing**: Process multiple URLs in one run rather than individual calls
2. **Parallel Processing**: For large URL lists, consider modifying the scraper to use threading
3. **Selective Scraping**: Focus on high-quality, authoritative sources
4. **Regular Updates**: Re-scrape content periodically to keep information current

## 🔗 Integration with Existing System

The web scraper is designed to work seamlessly with your existing PDF processing pipeline:

- **Same Output Format**: Creates compatible markdown chunks
- **Unified Search**: `embed_and_store_combined.py` merges all content
- **Source Tracking**: Maintains metadata about content origins
- **Consistent Interface**: RAG agent works the same way with mixed content

Your users won't know whether information came from PDFs or websites - they'll just get comprehensive, accurate answers from your expanded knowledge base! 

📄 PDFs (data/) ──┐
                   ├─► 📝 Chunks (output_all/) ──► 🗄️ Unified FAISS Index ──► 💬 RAG Agent
🌐 Websites ───────┘