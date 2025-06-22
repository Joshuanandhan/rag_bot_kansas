from chunkr_ai import Chunkr
from chunkr_ai.models import (
    Configuration,
    ChunkProcessing,
    SegmentProcessing,
    GenerationConfig,
    GenerationStrategy,
    LlmProcessing,
    FallbackStrategy,
    EmbedSource,
    Tokenizer
)
import openai
import json
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any
import os
import glob

def get_pdf_config():
    """
    Get the cost-effective configuration for PDF processing
    """
    return Configuration(
        # Chunking optimized for OpenAI
        chunk_processing=ChunkProcessing(
            target_length=1024,  # 1024 tokens as requested
            tokenizer=Tokenizer.CL100K_BASE,  # OpenAI's tokenizer
            ignore_headers_and_footers=True
        ),
        
        # Minimal LLM usage for cost effectiveness
        llm_processing=LlmProcessing(
            model_id="gpt-4o-mini",  # Most cost-effective model
            fallback_strategy=FallbackStrategy.model("gpt-3.5-turbo"),
            temperature=0.0
        ),
        
        # Only process tables with LLM, text uses fast heuristics
        segment_processing=SegmentProcessing(
            # Tables get LLM processing for better searchability
            Table=GenerationConfig(
                markdown=GenerationStrategy.LLM,
                llm="Convert this table to clear, structured text preserving all key data",
                embed_sources=[EmbedSource.LLM],
                extended_context=True
            ),
            
            # All text elements use fast AUTO processing (no LLM costs)
            Text=GenerationConfig(
                markdown=GenerationStrategy.AUTO,
                embed_sources=[EmbedSource.MARKDOWN]
            ),
            
            Title=GenerationConfig(
                markdown=GenerationStrategy.AUTO,
                embed_sources=[EmbedSource.MARKDOWN]
            ),
            
            SectionHeader=GenerationConfig(
                markdown=GenerationStrategy.AUTO,
                embed_sources=[EmbedSource.MARKDOWN]
            ),
            
            ListItem=GenerationConfig(
                markdown=GenerationStrategy.AUTO,
                embed_sources=[EmbedSource.MARKDOWN]
            )
        )
    )

def process_multiple_pdfs(folder_path, chunkr_api_key):
    """
    Process all PDF files in a folder
    """
    # Find all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder_path}")
    
    print(f"📁 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"   - {os.path.basename(pdf_file)}")
    
    # Initialize Chunkr client
    chunkr = Chunkr(api_key=chunkr_api_key)
    config = get_pdf_config()
    
    all_tasks = []
    all_chunks_data = []
    
    for i, pdf_path in enumerate(pdf_files):
        pdf_name = os.path.basename(pdf_path)
        print(f"\n📄 Processing {pdf_name} ({i+1}/{len(pdf_files)})...")
        
        try:
            # Process the PDF
            task = chunkr.upload(pdf_path, config)
            all_tasks.append(task)
            
            # Collect chunks with source information
            chunks = task.output.chunks
            print(f"   ✅ {pdf_name}: {len(chunks)} chunks created")
            
            # Add source information to each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_data = {
                    'source_file': pdf_name,
                    'source_path': pdf_path,
                    'file_index': i,
                    'chunk_index_in_file': chunk_idx,
                    'global_chunk_id': len(all_chunks_data),
                    'content': chunk.embed,
                    'segments': [
                        {
                            'type': seg.segment_type,
                            'content': seg.markdown if hasattr(seg, 'markdown') else '',
                            'llm_content': seg.llm if hasattr(seg, 'llm') and seg.llm else None
                        } for seg in chunk.segments
                    ],
                    'segment_types': [seg.segment_type for seg in chunk.segments],
                    'has_tables': any(seg.segment_type == "Table" for seg in chunk.segments),
                    'segment_count': len(chunk.segments),
                    'token_count': len(chunk.embed.split())  # Approximate
                }
                all_chunks_data.append(chunk_data)
                
        except Exception as e:
            print(f"   ❌ Error processing {pdf_name}: {str(e)}")
            continue
    
    # Clean up Chunkr connection
    chunkr.close()
    
    print(f"\n✅ Processed {len(all_tasks)} PDFs successfully")
    print(f"📊 Total chunks created: {len(all_chunks_data)}")
    
    # Count segments by type
    total_text_segments = 0
    total_table_segments = 0
    
    for chunk_data in all_chunks_data:
        for seg_type in chunk_data['segment_types']:
            if seg_type == "Table":
                total_table_segments += 1
            elif seg_type in ["Text", "Title", "SectionHeader", "ListItem"]:
                total_text_segments += 1
    
    print(f"📈 Total segments: {total_text_segments} text, {total_table_segments} tables")
    
    return all_tasks, all_chunks_data

def create_unified_faiss_index(chunks_data, openai_api_key, faiss_index_path="unified_index"):
    """
    Create a unified FAISS index from all PDF chunks
    """
    print(f"\n🔗 Creating unified FAISS index from {len(chunks_data)} chunks...")
    
    # Initialize OpenAI client
    openai_client = openai.OpenAI(api_key=openai_api_key)
    
    embeddings = []
    
    # Create embeddings for all chunks
    for i, chunk_data in enumerate(chunks_data):
        print(f"Creating embedding {i+1}/{len(chunks_data)} ({chunk_data['source_file']})...", end="\r")
        
        # Create embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",  # 1536 dimensions, cost-effective
            input=chunk_data['content']
        )
        
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    
    print(f"\n✅ Created {len(embeddings)} embeddings")
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]  # 1536 for text-embedding-3-small
    print(f"🔍 Creating unified FAISS index with dimension {dimension}")
    
    # Use IndexFlatIP for cosine similarity (after normalization)
    faiss.normalize_L2(embeddings_array)
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to index
    index.add(embeddings_array)
    
    print(f"📚 Added {index.ntotal} vectors to unified FAISS index")
    
    # Save FAISS index and metadata
    faiss.write_index(index, f"{faiss_index_path}.index")
    
    with open(f"{faiss_index_path}_metadata.pkl", "wb") as f:
        pickle.dump(chunks_data, f)
    
    with open(f"{faiss_index_path}_metadata.json", "w") as f:
        json.dump(chunks_data, f, indent=2)
    
    print(f"💾 Unified FAISS index saved to {faiss_index_path}.index")
    print(f"💾 Metadata saved to {faiss_index_path}_metadata.pkl and .json")
    
    return index, chunks_data, embeddings_array

class UnifiedFAISSSearch:
    """
    Class for searching the unified FAISS index across multiple PDFs
    """
    
    def __init__(self, index_path="unified_index", openai_api_key=None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Load FAISS index
        self.index = faiss.read_index(f"{index_path}.index")
        
        # Load metadata
        with open(f"{index_path}_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"🔍 Loaded unified FAISS index with {self.index.ntotal} vectors from {len(set(chunk['source_file'] for chunk in self.metadata))} PDFs")
    
    def search(self, query: str, k: int = 5, filter_by_file: str = None) -> List[Dict[Any, Any]]:
        """
        Search for similar chunks across all PDFs
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key required for search")
        
        # Create embedding for query
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        
        query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index (get more results if filtering)
        search_k = k * 3 if filter_by_file else k
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                chunk_data = self.metadata[idx]
                
                # Apply file filter if specified
                if filter_by_file and chunk_data['source_file'] != filter_by_file:
                    continue
                
                result = {
                    'score': float(score),
                    'source_file': chunk_data['source_file'],
                    'global_chunk_id': chunk_data['global_chunk_id'],
                    'chunk_index_in_file': chunk_data['chunk_index_in_file'],
                    'content': chunk_data['content'],
                    'segment_types': chunk_data['segment_types'],
                    'has_tables': chunk_data['has_tables'],
                    'token_count': chunk_data['token_count']
                }
                results.append(result)
                
                if len(results) >= k:
                    break
        
        # Add rank
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results
    
    def search_by_file(self, query: str, filename: str, k: int = 5) -> List[Dict[Any, Any]]:
        """
        Search within a specific PDF file
        """
        return self.search(query, k, filter_by_file=filename)
    
    def get_file_stats(self) -> Dict[str, Any]:
        """
        Get statistics about each file in the index
        """
        file_stats = {}
        
        for chunk in self.metadata:
            filename = chunk['source_file']
            if filename not in file_stats:
                file_stats[filename] = {
                    'chunk_count': 0,
                    'total_tokens': 0,
                    'has_tables': 0,
                    'segment_types': set()
                }
            
            file_stats[filename]['chunk_count'] += 1
            file_stats[filename]['total_tokens'] += chunk['token_count']
            if chunk['has_tables']:
                file_stats[filename]['has_tables'] += 1
            file_stats[filename]['segment_types'].update(chunk['segment_types'])
        
        # Convert sets to lists for JSON serialization
        for filename in file_stats:
            file_stats[filename]['segment_types'] = list(file_stats[filename]['segment_types'])
        
        return file_stats
    
    def list_files(self) -> List[str]:
        """
        Get list of all files in the index
        """
        return list(set(chunk['source_file'] for chunk in self.metadata))

def export_unified_content(all_tasks, chunks_data, base_filename="unified_documents"):
    """
    Export processed content from all PDFs
    """
    print(f"\n📁 Exporting unified content from {len(all_tasks)} PDFs...")
    
    # Export combined markdown
    combined_markdown = ""
    for i, task in enumerate(all_tasks):
        pdf_name = chunks_data[0]['source_file'] if chunks_data else f"document_{i+1}"
        # Find the actual PDF name for this task
        task_chunks = [c for c in chunks_data if c['file_index'] == i]
        if task_chunks:
            pdf_name = task_chunks[0]['source_file']
        
        combined_markdown += f"\n\n# {pdf_name}\n\n"
        combined_markdown += task.markdown()
        combined_markdown += "\n\n" + "="*80 + "\n\n"
    
    with open(f"{base_filename}.md", "w", encoding="utf-8") as f:
        f.write(combined_markdown)
    
    # Export chunk content with source information
    with open(f"{base_filename}_chunks.txt", "w", encoding="utf-8") as f:
        for chunk_data in chunks_data:
            f.write(f"=== CHUNK {chunk_data['global_chunk_id']} ===\n")
            f.write(f"Source: {chunk_data['source_file']}\n")
            f.write(f"Chunk in file: {chunk_data['chunk_index_in_file']}\n")
            f.write(f"Segments: {chunk_data['segment_types']}\n")
            f.write(f"Has tables: {chunk_data['has_tables']}\n")
            f.write(f"Content:\n{chunk_data['content']}\n")
            f.write("="*80 + "\n\n")
    
    print(f"✅ Unified content exported to {base_filename}.md and {base_filename}_chunks.txt")

def demo_unified_search(search_engine: UnifiedFAISSSearch):
    """
    Demo function to show unified search capabilities
    """
    print("\n" + "="*60)
    print("🔍 DEMO: Searching across all your PDFs")
    print("="*60)
    
    # Show file statistics
    file_stats = search_engine.get_file_stats()
    print("\n📊 File Statistics:")
    for filename, stats in file_stats.items():
        print(f"   📄 {filename}:")
        print(f"      - Chunks: {stats['chunk_count']}")
        print(f"      - Tokens: {stats['total_tokens']:,}")
        print(f"      - Tables: {stats['has_tables']}")
        print(f"      - Segment types: {', '.join(stats['segment_types'])}")
    
    # Example searches
    demo_queries = [
        "What are the main findings across all documents?",
        "Show me information about tables and data",
        "What are the key methodologies discussed?",
        "Tell me about regulatory requirements",
        "What conclusions are drawn in these documents?"
    ]
    
    print(f"\n🔍 Running {len(demo_queries)} demo searches:")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n--- Demo Search {i} ---")
        print(f"Query: {query}")
        
        try:
            results = search_engine.search(query, k=3)
            
            if results:
                print(f"Found {len(results)} relevant chunks:")
                for result in results:
                    print(f"   📄 {result['source_file']} (Chunk {result['chunk_index_in_file']})")
                    print(f"      Score: {result['score']:.4f}")
                    print(f"      Types: {', '.join(result['segment_types'])}")
                    print(f"      Preview: {result['content'][:100]}...")
                    print()
            else:
                print("   No results found.")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ Demo complete! You can now use the search_engine object for your own queries.")
    print("="*60)

def interactive_search(search_engine: UnifiedFAISSSearch):
    """
    Interactive search interface for querying the unified index
    """
    print("\n" + "="*60)
    print("🔍 INTERACTIVE SEARCH MODE")
    print("="*60)
    print("Type your queries (or 'quit' to exit, 'files' to list available files)")
    
    while True:
        try:
            query = input("\nEnter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'files':
                files = search_engine.list_files()
                print(f"\n📁 Available files ({len(files)}):")
                for i, filename in enumerate(files, 1):
                    print(f"   {i}. {filename}")
                continue
            elif not query:
                continue
            
            # Check if user wants to search within a specific file
            if query.startswith("file:"):
                parts = query.split(":", 2)
                if len(parts) >= 3:
                    filename = parts[1].strip()
                    actual_query = parts[2].strip()
                    print(f"🔍 Searching in '{filename}' for: {actual_query}")
                    results = search_engine.search_by_file(actual_query, filename, k=5)
                else:
                    print("❌ Format: file:filename:your query")
                    continue
            else:
                print(f"🔍 Searching across all files for: {query}")
                results = search_engine.search(query, k=5)
            
            if results:
                print(f"\n📊 Found {len(results)} relevant chunks:")
                for result in results:
                    print(f"\n   #{result['rank']} - {result['source_file']} (Chunk {result['chunk_index_in_file']})")
                    print(f"      Score: {result['score']:.4f}")
                    print(f"      Types: {', '.join(result['segment_types'])}")
                    if result['has_tables']:
                        print("      ⚠️  Contains tables")
                    print(f"      Content: {result['content'][:200]}...")
            else:
                print("   ❌ No results found. Try a different query.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """
    Main function to orchestrate the entire PDF processing and search workflow
    """
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    print("🚀 PDF Multi-Document Processing and Search System")
    print("="*60)
    
    # Configuration
    CHUNKR_API_KEY = os.getenv("CHUNKR_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PDF_FOLDER = os.getenv("PDF_FOLDER", "./data")  # Default to ./data folder
    INDEX_NAME = os.getenv("INDEX_NAME", "unified_index")
    
    if not CHUNKR_API_KEY:
        print("❌ Error: CHUNKR_API_KEY environment variable not set")
        return
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    print(f"📁 PDF Folder: {PDF_FOLDER}")
    print(f"🔍 Index Name: {INDEX_NAME}")
    
    try:
        # Check if index already exists
        index_exists = (
            os.path.exists(f"{INDEX_NAME}.index") and 
            os.path.exists(f"{INDEX_NAME}_metadata.pkl")
        )
        
        if index_exists:
            print(f"\n✅ Found existing index: {INDEX_NAME}")
            choice = input("Use existing index? (y/n): ").strip().lower()
            
            if choice == 'y':
                # Load existing index
                search_engine = UnifiedFAISSSearch(INDEX_NAME, OPENAI_API_KEY)
                print("✅ Loaded existing search index")
            else:
                # Process PDFs and create new index
                print("\n🔄 Processing PDFs and creating new index...")
                all_tasks, chunks_data = process_multiple_pdfs(PDF_FOLDER, CHUNKR_API_KEY)
                
                if not chunks_data:
                    print("❌ No chunks were created. Exiting.")
                    return
                
                # Create FAISS index
                index, chunks_data, embeddings = create_unified_faiss_index(
                    chunks_data, OPENAI_API_KEY, INDEX_NAME
                )
                
                # Export content
                export_unified_content(all_tasks, chunks_data, "unified_documents")
                
                # Create search engine
                search_engine = UnifiedFAISSSearch(INDEX_NAME, OPENAI_API_KEY)
        else:
            # Process PDFs and create index
            print("\n🔄 Processing PDFs and creating search index...")
            all_tasks, chunks_data = process_multiple_pdfs(PDF_FOLDER, CHUNKR_API_KEY)
            
            if not chunks_data:
                print("❌ No chunks were created. Exiting.")
                return
            
            # Create FAISS index
            index, chunks_data, embeddings = create_unified_faiss_index(
                chunks_data, OPENAI_API_KEY, INDEX_NAME
            )
            
            # Export content
            export_unified_content(all_tasks, chunks_data, "unified_documents")
            
            # Create search engine
            search_engine = UnifiedFAISSSearch(INDEX_NAME, OPENAI_API_KEY)
        
        # Run demo
        demo_unified_search(search_engine)
        
        # Interactive search mode
        interactive_choice = input("\nStart interactive search mode? (y/n): ").strip().lower()
        if interactive_choice == 'y':
            interactive_search(search_engine)
        
        print("\n✅ Processing complete!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()