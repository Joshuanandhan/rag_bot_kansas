from chunkr_ai import Chunkr
from chunkr_ai.models import (
    Configuration,
    ChunkProcessing,
    Tokenizer
)

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

def process_pdf_for_rag(pdf_path, chunkr_api_key):
    """
    Process PDF with intelligent chunking
    Cost-effective setup focusing on text and tables
    """
    
    # Basic chunking configuration
    config = Configuration(
        chunk_processing=ChunkProcessing(
            target_length=1024,  # 1024 tokens as requested
            tokenizer=Tokenizer.CL100K_BASE,  # OpenAI's tokenizer
            ignore_headers_and_footers=True
        )
    )
    
    # Initialize Chunkr client
    print(f"🚀 Processing PDF: {pdf_path}")
    chunkr = Chunkr(api_key=chunkr_api_key)
    
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

def save_chunks(task, output_dir, file_basename):
    """
    Save chunks in the specified format:
    {output_dir}/{file_basename}/chunks.json
    {output_dir}/{file_basename}/chunks/chunk_{i}.md
    """
    print(f"💾 Saving chunks for {file_basename}...")
    
    # Create output directory structure
    base_output_dir = Path(output_dir) / file_basename
    chunks_dir = base_output_dir / "chunks"
    
    base_output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    chunks = task.output.chunks
    
    # Prepare chunks data for JSON
    chunks_data = []
    
    for i, chunk in enumerate(chunks):
        # Save individual chunk as markdown
        chunk_file = chunks_dir / f"chunk_{i}.md"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            # f.write(f"# Chunk {i}\n\n")
            # f.write(f"**Segment Types:** {[seg.segment_type for seg in chunk.segments]}\n\n")
            # f.write(f"**Segment Count:** {len(chunk.segments)}\n\n")
            # f.write("---\n\n")
            f.write(chunk.embed)
        
        # Prepare chunk data for JSON
        chunk_data = {
            'chunk_id': i,
            'content': chunk.embed,
            'token_count': len(chunk.embed.split()),  # Approximate
            'segment_types': [seg.segment_type for seg in chunk.segments],
            'has_tables': any(seg.segment_type == "Table" for seg in chunk.segments),
            'segment_count': len(chunk.segments),
            'file_path': f"chunks/chunk_{i}.md"
        }
        chunks_data.append(chunk_data)
    
    # Save chunks.json
    chunks_json_file = base_output_dir / "chunks.json"
    json_data = task.json(output_file=chunks_json_file)    
    print(f"✅ Saved {len(chunks)} chunks to {base_output_dir}")
    print(f"   - chunks.json: {chunks_json_file}")
    print(f"   - Individual chunks: {chunks_dir}/chunk_*.md")
    
    return len(chunks)

def get_pdf_files(path):
    """Get list of PDF files from path (file or directory)"""
    path = Path(path)
    
    if path.is_file():
        if path.suffix.lower() == '.pdf':
            return [path]
        else:
            raise ValueError(f"File {path} is not a PDF")
    elif path.is_dir():
        pdf_files = list(path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory {path}")
        return pdf_files
    else:
        raise ValueError(f"Path {path} does not exist")

def main():
    """
    Main function - Complete workflow with command line arguments
    """
    parser = argparse.ArgumentParser(description="Process PDF files with Chunkr AI")
    
    # Create mutually exclusive group for file/directory input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", type=str, help="Path to a single PDF file to process")
    input_group.add_argument("-d", "--dir", type=str, help="Directory containing PDF files to process")
    
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory for processed chunks")
    
    args = parser.parse_args()
    
    # Get API key
    CHUNKR_API_KEY = os.getenv("CHUNKR_API_KEY")
    
    if not CHUNKR_API_KEY:
        print("❌ Error: CHUNKR_API_KEY not found in .env file")
        return
    
    print("✅ API key loaded from .env file")
    
    # Determine input path
    input_path = args.file if args.file else args.dir
    
    try:
        # Get list of PDF files to process
        pdf_files = get_pdf_files(input_path)
        print(f"📁 Found {len(pdf_files)} PDF file(s) to process")
        
        total_chunks = 0
        processed_files = 0
        
        for pdf_file in pdf_files:
            try:
                print(f"\n{'='*60}")
                print(f"Processing: {pdf_file.name}")
                print(f"{'='*60}")
                
                # Check file size
                file_size = pdf_file.stat().st_size / (1024 * 1024)  # Size in MB
                print(f"📊 File size: {file_size:.2f} MB")
                
                # Process the PDF
                task = process_pdf_for_rag(str(pdf_file), CHUNKR_API_KEY)
                
                # Save chunks
                file_basename = pdf_file.stem  # filename without extension
                chunks_count = save_chunks(task, args.output, file_basename)
                
                total_chunks += chunks_count
                processed_files += 1
                
                print(f"✅ {pdf_file.name} processed successfully ({chunks_count} chunks)")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {str(e)}")
                continue
        
        # Final summary
        print(f"\n{'='*60}")
        print("📋 PROCESSING COMPLETE - SUMMARY")
        print(f"{'='*60}")
        print(f"📄 Files processed: {processed_files}/{len(pdf_files)}")
        print(f"📊 Total chunks created: {total_chunks}")
        print(f"📁 Output directory: {args.output}")
        
        if processed_files > 0:
            print(f"🎯 Average chunks per file: {total_chunks // processed_files}")
            print("\n✅ Your PDF files are now ready for RAG!")
            print(f"📁 Check output in: {args.output}/")
        
    except ValueError as e:
        print(f"❌ Error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("Please check your API key and file paths")

if __name__ == "__main__":
    main()