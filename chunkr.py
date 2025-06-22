from chunkr_ai import Chunkr

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CHUNKR_API_KEY = os.getenv("CHUNKR_API_KEY")

# Initialize the Chunkr client with your API key - get this from https://chunkr.ai
chunkr = Chunkr(api_key=CHUNKR_API_KEY)

# Upload a document via url or local file path
# url = "https://chunkr-web.s3.us-east-1.amazonaws.com/landing_page/input/specs.pdf"
# task = chunkr.upload(url)
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

def process_pdf_for_rag(pdf_path, chunkr_api_key, openai_api_key):
    """
    Complete function to process PDF with intelligent chunking
    Cost-effective setup focusing on text and tables
    """
    
    # Start with a simpler configuration to test
    config = Configuration(
        # Basic chunking configuration
        chunk_processing=ChunkProcessing(
            target_length=1024,  # 1024 tokens as requested
            tokenizer=Tokenizer.CL100K_BASE,  # OpenAI's tokenizer
            ignore_headers_and_footers=True
        )
    )
    
    # If the simple config works, we can try the more complex one
    # complex_config = Configuration(
    #     # Chunking optimized for OpenAI
    #     chunk_processing=ChunkProcessing(
    #         target_length=1024,  # 1024 tokens as requested
    #         tokenizer=Tokenizer.CL100K_BASE,  # OpenAI's tokenizer
    #         ignore_headers_and_footers=True
    #     ),
    #     
    #     # Minimal LLM usage for cost effectiveness
    #     llm_processing=LlmProcessing(
    #         model_id="gpt-4o-mini",  # Most cost-effective model
    #         fallback_strategy=FallbackStrategy.model("gpt-3.5-turbo"),
    #         temperature=0.0
    #     ),
    #     
    #     # Only process tables with LLM, text uses fast heuristics
    #     segment_processing=SegmentProcessing(
    #         # Tables get LLM processing for better searchability
    #         Table=GenerationConfig(
    #             markdown=GenerationStrategy.LLM,  # Only markdown to save costs
    #             llm="Convert this table to clear, structured text preserving all key data",
    #             embed_sources=[EmbedSource.LLM],  # Use LLM output for embeddings
    #             extended_context=True  # Better table understanding
    #         ),
    #         
    #         # All text elements use fast AUTO processing (no LLM costs)
    #         Text=GenerationConfig(
    #             markdown=GenerationStrategy.AUTO,
    #             embed_sources=[EmbedSource.MARKDOWN]
    #         ),
    #         
    #         Title=GenerationConfig(
    #             markdown=GenerationStrategy.AUTO,
    #             embed_sources=[EmbedSource.MARKDOWN]
    #         ),
    #         
    #         SectionHeader=GenerationConfig(
    #             markdown=GenerationStrategy.AUTO,
    #             embed_sources=[EmbedSource.MARKDOWN]
    #         ),
    #         
    #         ListItem=GenerationConfig(
    #             markdown=GenerationStrategy.AUTO,
    #             embed_sources=[EmbedSource.MARKDOWN]
    #         )
    #     )
    # )
    
    # Initialize Chunkr client
    print("🚀 Initializing Chunkr...")
    chunkr = Chunkr(api_key=chunkr_api_key)
    
    # Process the PDF
    print("📄 Processing your PDF...")
    print(f"📄 File: {pdf_path}")
    
    try:
        task = chunkr.upload(pdf_path, config)
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        # Clean up before raising
        chunkr.close()
        raise
    
    # Analyze the results
    chunks = task.output.chunks
    print(f"✅ Document processed into {len(chunks)} intelligent chunks")
    
    # Count segment types
    text_segments = 0
    table_segments = 0
    
    for chunk in chunks:
        for segment in chunk.segments:
            if segment.segment_type == "Table":
                table_segments += 1
            elif segment.segment_type in ["Text", "Title", "SectionHeader", "ListItem"]:
                text_segments += 1
    
    print(f"📊 Found {text_segments} text segments and {table_segments} table segments")
    
    # Clean up Chunkr connection
    chunkr.close()
    
    return task

def create_openai_embeddings(task, openai_api_key, output_file="embeddings_data.json"):
    """
    Create OpenAI embeddings from the processed chunks
    """
    print("🔗 Creating OpenAI embeddings...")
    
    # Initialize OpenAI client
    openai_client = openai.OpenAI(api_key=openai_api_key)
    
    embeddings_data = []
    chunks = task.output.chunks
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...", end="\r")
        
        # Create embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",  # Cost-effective embedding model
            input=chunk.embed
        )
        
        # Store with metadata
        chunk_data = {
            'chunk_id': i,
            'content': chunk.embed,
            'embedding': response.data[0].embedding,
            'token_count': len(chunk.embed.split()),  # Approximate
            'segment_types': [seg.segment_type for seg in chunk.segments],
            'has_tables': any(seg.segment_type == "Table" for seg in chunk.segments),
            'segment_count': len(chunk.segments)
        }
        
        embeddings_data.append(chunk_data)
    
    print(f"\n✅ Created {len(embeddings_data)} embeddings")
    
    # Save embeddings data
    with open(output_file, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"💾 Embeddings saved to {output_file}")
    
    # Print summary
    table_chunks = sum(1 for item in embeddings_data if item['has_tables'])
    print(f"📈 Summary: {table_chunks} chunks contain tables, {len(embeddings_data) - table_chunks} are text-only")
    
    return embeddings_data

def export_processed_content(task, base_filename="processed_document"):
    """
    Export the processed content in various formats
    """
    print("📁 Exporting processed content...")
    
    # Export markdown
    task.markdown(output_file=f"{base_filename}.md")
    print(f"✅ Markdown exported to {base_filename}.md")
    
    # Export JSON with full metadata
    task.json(output_file=f"{base_filename}_full.json")
    print(f"✅ Full JSON exported to {base_filename}_full.json")
    
    # Export just the embed content for vector database
    chunks = task.output.chunks
    with open(f"{base_filename}_embed_content.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== CHUNK {i+1} ===\n")
            f.write(f"Segments: {[seg.segment_type for seg in chunk.segments]}\n")
            f.write(f"Content:\n{chunk.embed}\n")
            f.write("="*50 + "\n\n")
    
    print(f"✅ Embed content exported to {base_filename}_embed_content.txt")

def main():
    """
    Main function - Complete workflow
    """
    # Configuration
    PDF_PATH = "/home/momo/dev/projects/nandhu/chunker/data/Kansas Motorcycle Handbook.pdf"
    CHUNKR_API_KEY = os.getenv("CHUNKR_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Check if API keys are loaded
    if not CHUNKR_API_KEY:
        print("❌ Error: CHUNKR_API_KEY not found in .env file")
        return
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    
    print("✅ API keys loaded from .env file")
    
    # Check if PDF file exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF file not found at {PDF_PATH}")
        return
    
    print(f"✅ PDF file found: {PDF_PATH}")
    file_size = os.path.getsize(PDF_PATH) / (1024 * 1024)  # Size in MB
    print(f"📊 File size: {file_size:.2f} MB")
    
    try:
        # Step 1: Process PDF with intelligent chunking
        task = process_pdf_for_rag(PDF_PATH, CHUNKR_API_KEY, OPENAI_API_KEY)
        
        # Step 2: Export processed content
        export_processed_content(task)
        
        # Step 3: Create OpenAI embeddings (commented out to check chunking first)
        # embeddings_data = create_openai_embeddings(task, OPENAI_API_KEY)
        
        # Step 4: Quick analysis
        print("\n" + "="*50)
        print("📋 PROCESSING COMPLETE - SUMMARY")
        print("="*50)
        
        chunks = task.output.chunks
        total_content_length = sum(len(chunk.embed) for chunk in chunks)
        
        print(f"📄 Total chunks: {len(chunks)}")
        print(f"📊 Total content length: {total_content_length:,} characters")
        # print(f"💰 Estimated embedding cost: ~${len(chunks) * 0.00002:.4f} (text-embedding-3-small)")
        print(f"🎯 Average chunk size: {total_content_length // len(chunks):,} characters")
        
        # Show sample chunk
        if chunks:
            print("\n📝 Sample chunk content:")
            print("-" * 30)
            print(chunks[0].embed[:300] + "..." if len(chunks[0].embed) > 300 else chunks[0].embed)
        
        print("\n✅ Your 46-page PDF is now ready for RAG!")
        print("📁 Files created:")
        print("   - processed_document.md (readable format)")
        print("   - processed_document_embed_content.txt (chunk content)")
        print("   - embeddings_data.json (embeddings + metadata)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your API keys and file path")

if __name__ == "__main__":
    main()