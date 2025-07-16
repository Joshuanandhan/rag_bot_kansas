"""
Combined Embedding and Storage Script
Processes both PDF chunks and web scraped content into a unified FAISS index
"""

import dotenv
import os
from pathlib import Path
import json

# Load environment variables from .env file
dotenv.load_dotenv()

from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

import tiktoken
import time
from typing import List, Dict

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string using tiktoken
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    try:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

def batch_documents_by_tokens(documents: list, max_tokens: int = 250000) -> List[List]:
    """
    Split documents into batches that don't exceed token limit
    
    Args:
        documents: List of Document objects
        max_tokens: Maximum tokens per batch (with safety margin)
        
    Returns:
        List of document batches
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    print(f"📊 Batching documents (max {max_tokens:,} tokens per batch)...")
    
    for i, doc in enumerate(documents):
        doc_tokens = estimate_tokens(doc.page_content)
        
        # If adding this document would exceed limit, start new batch
        if current_tokens + doc_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            print(f"   Batch {len(batches)}: {len(current_batch)} documents ({current_tokens:,} tokens)")
            current_batch = [doc]
            current_tokens = doc_tokens
        else:
            current_batch.append(doc)
            current_tokens += doc_tokens
    
    # Add the last batch
    if current_batch:
        batches.append(current_batch)
        print(f"   Batch {len(batches)}: {len(current_batch)} documents ({current_tokens:,} tokens)")
    
    print(f"✅ Created {len(batches)} batches")
    return batches

def create_unified_faiss_index(documents: list, index_name: str = "unified_faiss_index") -> FAISS:
    """
    Create a unified FAISS index from all documents with batch processing
    
    Args:
        documents: List of Document objects
        index_name: Name for the FAISS index
        
    Returns:
        FAISS vector store
    """
    if not documents:
        print("❌ No documents to process")
        return None
    
    print(f"🔄 Creating embeddings for {len(documents)} documents...")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Get embedding dimension
    print("📐 Testing embedding dimension...")
    embedding_dim = len(embeddings.embed_query("hello world"))
    print(f"📐 Embedding dimension: {embedding_dim}")
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Create vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Calculate total tokens
    total_tokens = sum(estimate_tokens(doc.page_content) for doc in documents)
    print(f"📊 Total estimated tokens: {total_tokens:,}")
    
    # Batch documents by token count
    batches = batch_documents_by_tokens(documents, max_tokens=250000)  # Safety margin
    
    # Process batches
    total_added = 0
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n🔄 Processing batch {batch_num}/{len(batches)} ({len(batch)} documents)...")
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Add documents to vector store
                ids = vector_store.add_documents(documents=batch)
                total_added += len(ids)
                print(f"✅ Batch {batch_num} processed successfully ({len(ids)} documents)")
                break
                
            except Exception as e:
                retry_count += 1
                if "max_tokens_per_request" in str(e) or "400" in str(e):
                    print(f"⚠️  Token limit exceeded in batch {batch_num}, retrying with smaller batch...")
                    # Split the batch in half and retry
                    if len(batch) > 1:
                        mid = len(batch) // 2
                        batch1 = batch[:mid]
                        batch2 = batch[mid:]
                        
                        # Process first half
                        try:
                            ids1 = vector_store.add_documents(documents=batch1)
                            total_added += len(ids1)
                            print(f"✅ Batch {batch_num}a processed ({len(ids1)} documents)")
                        except Exception as e1:
                            print(f"❌ Error in batch {batch_num}a: {e1}")
                        
                        # Process second half
                        try:
                            ids2 = vector_store.add_documents(documents=batch2)
                            total_added += len(ids2)
                            print(f"✅ Batch {batch_num}b processed ({len(ids2)} documents)")
                        except Exception as e2:
                            print(f"❌ Error in batch {batch_num}b: {e2}")
                        
                        break
                    else:
                        print(f"❌ Single document in batch {batch_num} is too large: {e}")
                        break
                else:
                    print(f"❌ Error in batch {batch_num} (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        print(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Failed to process batch {batch_num} after {max_retries} attempts")
                        break
        
        # Small delay between batches to be respectful to the API
        if batch_num < len(batches):
            time.sleep(1)
    
    print(f"\n✅ Added {total_added} documents to vector store")
    
    # Save the vector store
    print(f"💾 Saving vector store as '{index_name}'...")
    vector_store.save_local(index_name)
    print(f"✅ Vector store saved successfully")
    
    return vector_store

def load_all_content(output_dir: str = "output_all") -> list:
    """
    Load all content from both PDF chunks and web scraped content
    
    Args:
        output_dir: Directory containing processed content
        
    Returns:
        List of Document objects
    """
    all_documents = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Output directory {output_dir} does not exist")
        return []
    
    # Find all subdirectories (each represents a source)
    source_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    
    print(f"📁 Found {len(source_dirs)} content sources:")
    
    for source_dir in source_dirs:
        print(f"   - {source_dir.name}")
        
        # Load markdown files from chunks directory
        chunks_dir = source_dir / "chunks"
        if chunks_dir.exists():
            try:
                # Load all markdown files
                loader = DirectoryLoader(
                    str(chunks_dir), 
                    glob="**/*.md", 
                    show_progress=True
                )
                docs = loader.load()
                
                # Add source information to metadata
                for doc in docs:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['source_type'] = source_dir.name
                    doc.metadata['content_source'] = str(source_dir)
                
                all_documents.extend(docs)
                print(f"   ✅ Loaded {len(docs)} chunks from {source_dir.name}")
                
            except Exception as e:
                print(f"   ❌ Error loading from {source_dir.name}: {e}")
                continue
        else:
            print(f"   ⚠️  No chunks directory found in {source_dir.name}")
    
    print(f"\n📊 Total documents loaded: {len(all_documents)}")
    return all_documents

def analyze_content_sources(documents: list):
    """
    Analyze and display statistics about content sources
    
    Args:
        documents: List of Document objects
    """
    if not documents:
        return
    
    # Count by source type
    source_counts = {}
    total_chars = 0
    total_words = 0
    
    for doc in documents:
        source_type = doc.metadata.get('source_type', 'unknown')
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        content = doc.page_content
        total_chars += len(content)
        total_words += len(content.split())
    
    print(f"\n📊 CONTENT ANALYSIS")
    print(f"{'='*50}")
    print(f"📄 Total documents: {len(documents)}")
    print(f"📝 Total words: {total_words:,}")
    print(f"🔤 Total characters: {total_chars:,}")
    print(f"📊 Average words per document: {total_words // len(documents):,}")
    
    print(f"\n📁 Content by source:")
    for source_type, count in source_counts.items():
        percentage = (count / len(documents)) * 100
        print(f"   - {source_type}: {count} documents ({percentage:.1f}%)")

def test_unified_search(vector_store: FAISS, test_queries: list = None):
    """
    Test the unified search with sample queries
    
    Args:
        vector_store: FAISS vector store
        test_queries: List of test queries
    """
    if not vector_store:
        return
    
    if test_queries is None:
        test_queries = [
            "commercial driver's license requirements",
            "motorcycle safety gear requirements", 
            "vehicle registration process",
            "Kansas driving laws",
            "speed limits on highways"
        ]
    
    print(f"\n🧪 TESTING UNIFIED SEARCH")
    print(f"{'='*50}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        print("-" * 40)
        
        try:
            results = vector_store.similarity_search(query, k=3)
            
            for j, result in enumerate(results, 1):
                source_type = result.metadata.get('source_type', 'unknown')
                content_preview = result.page_content[:200].replace('\n', ' ') + "..."
                
                print(f"   {j}. [{source_type}] {content_preview}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")

def create_content_summary(output_dir: str = "output_all", index_name: str = "unified_faiss_index"):
    """
    Create a summary file of all processed content
    
    Args:
        output_dir: Directory containing processed content
        index_name: Name of the FAISS index
    """
    output_path = Path(output_dir)
    summary_file = Path(f"{index_name}_content_summary.md")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# RAG System Content Summary\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Index Name:** {index_name}\n\n")
        
        # List all content sources
        if output_path.exists():
            source_dirs = [d for d in output_path.iterdir() if d.is_dir()]
            
            f.write("## Content Sources\n\n")
            for source_dir in source_dirs:
                f.write(f"### {source_dir.name}\n")
                
                # Check for summary files
                summary_files = list(source_dir.glob("*summary*"))
                chunks_dir = source_dir / "chunks"
                
                if chunks_dir.exists():
                    chunk_files = list(chunks_dir.glob("*.md"))
                    f.write(f"- **Chunks:** {len(chunk_files)}\n")
                
                if summary_files:
                    f.write(f"- **Summary:** {summary_files[0].name}\n")
                
                f.write("\n")
        
        f.write("## Usage Instructions\n\n")
        f.write("```python\n")
        f.write("from langchain_community.vectorstores import FAISS\n")
        f.write("from langchain_openai import OpenAIEmbeddings\n\n")
        f.write("# Load the unified index\n")
        f.write("embeddings = OpenAIEmbeddings(model='text-embedding-3-large')\n")
        f.write(f"vector_store = FAISS.load_local('{index_name}', embeddings, allow_dangerous_deserialization=True)\n\n")
        f.write("# Search across all content\n")
        f.write("results = vector_store.similarity_search('your query here', k=5)\n")
        f.write("```\n")
    
    print(f"📄 Content summary saved to: {summary_file}")

def main():
    """
    Main function - Create unified embeddings from all content sources
    """
    print("🚀 Starting Combined Embedding and Storage Process")
    print("="*60)
    
    # Configuration
    output_dir = "output_all"
    index_name = "unified_faiss_index"
    
    # Load all content
    print("📂 Loading all content sources...")
    documents = load_all_content(output_dir)
    
    if not documents:
        print("❌ No documents found. Please run PDF processing and/or web scraping first.")
        print("\nTo get started:")
        print("1. Process PDFs: python chunkr.py -d data -o output_all")
        print("2. Scrape websites: python webscraper.py -f sample_urls.txt -o output_all -n web_content")
        print("3. Then run this script again")
        return
    
    # Analyze content
    analyze_content_sources(documents)
    
    # Create unified FAISS index
    print(f"\n🔄 Creating unified FAISS index...")
    vector_store = create_unified_faiss_index(documents, index_name)
    
    if vector_store:
        # Test the search
        test_unified_search(vector_store)
        
        # Create summary
        create_content_summary(output_dir, index_name)
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Unified index saved as: {index_name}")
        print(f"🎯 Ready for RAG! Your agent can now search across:")
        print(f"   - PDF content")
        print(f"   - Web scraped content") 
        print(f"   - Any future content sources")
        
        print(f"\n🔄 Next steps:")
        print(f"1. Update your agent.py to use '{index_name}' instead of 'faiss_index'")
        print(f"2. Run: python gradio_app.py")
    else:
        print("❌ Failed to create unified index")

if __name__ == "__main__":
    from datetime import datetime
    main() 