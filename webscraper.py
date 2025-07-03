"""
Web Scraper for RAG System Integration
Scrapes websites and processes content for RAG system
"""

import requests
from bs4 import BeautifulSoup
import os
import json
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
import re
from typing import List, Dict, Optional
import hashlib
from datetime import datetime

# Load environment variables
load_dotenv(override=True)

class WebScraper:
    def __init__(self, delay: float = 1.0, max_retries: int = 3):
        """
        Initialize the web scraper with respectful defaults
        
        Args:
            delay: Delay between requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set a respectful user agent
        self.session.headers.update({
            'User-Agent': 'RAG-Bot-WebScraper/1.0 (Educational/Research Purpose)'
        })
        
        # Track processed URLs to avoid duplicates
        self.processed_urls = set()
        
    def check_robots_txt(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt
        
        Args:
            url: URL to check
            
        Returns:
            True if scraping is allowed, False otherwise
        """
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch(self.session.headers['User-Agent'], url)
        except Exception as e:
            print(f"⚠️  Could not check robots.txt for {url}: {e}")
            return True  # Default to allowing if we can't check
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove very short lines (likely navigation, etc.)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 20:  # Only keep substantial lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_content(self, url: str) -> Optional[Dict]:
        """
        Extract content from a single URL
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with extracted content or None if failed
        """
        # Check if already processed
        if url in self.processed_urls:
            print(f"⏭️  Skipping already processed URL: {url}")
            return None
        
        # Check robots.txt
        if not self.check_robots_txt(url):
            print(f"🚫 Robots.txt disallows scraping: {url}")
            return None
        
        print(f"🌐 Scraping: {url}")
        
        for attempt in range(self.max_retries):
            try:
                # Add delay to be respectful
                if self.processed_urls:  # Don't delay on first request
                    time.sleep(self.delay)
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()
                
                # Extract title
                title = ""
                if soup.title:
                    title = soup.title.string.strip() if soup.title.string else ""
                
                # Extract main content
                # Try to find main content areas first
                content_selectors = [
                    'main', 'article', '.content', '#content', 
                    '.main-content', '#main-content', '.post-content',
                    '.entry-content', '.article-content'
                ]
                
                content_text = ""
                for selector in content_selectors:
                    content_elements = soup.select(selector)
                    if content_elements:
                        content_text = content_elements[0].get_text()
                        break
                
                # Fallback to body if no main content found
                if not content_text and soup.body:
                    content_text = soup.body.get_text()
                
                # Clean the text
                content_text = self.clean_text(content_text)
                
                # Extract metadata
                description = ""
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    description = meta_desc.get('content', '')
                
                # Create URL hash for unique identification
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                
                self.processed_urls.add(url)
                
                result = {
                    'url': url,
                    'title': title,
                    'content': content_text,
                    'description': description,
                    'scraped_at': datetime.now().isoformat(),
                    'word_count': len(content_text.split()),
                    'char_count': len(content_text),
                    'url_hash': url_hash,
                    'domain': urlparse(url).netloc
                }
                
                print(f"✅ Successfully scraped: {title} ({len(content_text)} chars)")
                return result
                
            except requests.RequestException as e:
                print(f"❌ Request failed for {url} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"❌ Failed to scrape {url} after {self.max_retries} attempts")
                    return None
            except Exception as e:
                print(f"❌ Unexpected error scraping {url}: {e}")
                return None
        
        return None
    
    def scrape_urls(self, urls: List[str]) -> List[Dict]:
        """
        Scrape multiple URLs
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped content dictionaries
        """
        results = []
        
        print(f"🚀 Starting to scrape {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            print(f"\n📍 Progress: {i}/{len(urls)}")
            
            content = self.extract_content(url)
            if content:
                results.append(content)
        
        return results

def save_scraped_content(scraped_data: List[Dict], output_dir: str, source_name: str = "web_content"):
    """
    Save scraped content in format compatible with existing RAG pipeline
    
    Args:
        scraped_data: List of scraped content dictionaries
        output_dir: Output directory path
        source_name: Name for the content source
    """
    print(f"💾 Saving scraped content...")
    
    # Create output directory structure
    base_output_dir = Path(output_dir) / source_name
    chunks_dir = base_output_dir / "chunks"
    
    base_output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each scraped page as chunks
    all_chunks_data = []
    chunk_id = 0
    
    for page_data in scraped_data:
        content = page_data['content']
        title = page_data['title']
        url = page_data['url']
        
        # Split content into chunks (roughly 1000 words per chunk to match PDF processing)
        words = content.split()
        chunk_size = 1000
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            # Skip very small chunks
            if len(chunk_content.strip()) < 100:
                continue
            
            # Create chunk markdown file
            chunk_file = chunks_dir / f"chunk_{chunk_id}.md"
            
            # Format content with metadata
            chunk_markdown = f"""# {title}

**Source URL:** {url}
**Domain:** {page_data['domain']}
**Scraped:** {page_data['scraped_at']}

---

{chunk_content}
"""
            
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk_markdown)
            
            # Prepare chunk metadata
            chunk_data = {
                'chunk_id': chunk_id,
                'content': chunk_markdown,
                'plain_content': chunk_content,
                'title': title,
                'source_url': url,
                'domain': page_data['domain'],
                'scraped_at': page_data['scraped_at'],
                'word_count': len(chunk_words),
                'char_count': len(chunk_content),
                'chunk_index': i // chunk_size,
                'total_chunks_for_page': (len(words) + chunk_size - 1) // chunk_size,
                'file_path': f"chunks/chunk_{chunk_id}.md"
            }
            
            all_chunks_data.append(chunk_data)
            chunk_id += 1
    
    # Save chunks metadata as JSON
    chunks_json_file = base_output_dir / "chunks.json"
    with open(chunks_json_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks_data, f, indent=2, ensure_ascii=False)
    
    # Save original scraped data
    original_data_file = base_output_dir / "original_scraped_data.json"
    with open(original_data_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary_file = base_output_dir / "scraping_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"# Web Scraping Summary\n\n")
        f.write(f"**Scraping Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Total Pages Scraped:** {len(scraped_data)}\n")
        f.write(f"**Total Chunks Created:** {len(all_chunks_data)}\n")
        f.write(f"**Total Word Count:** {sum(chunk['word_count'] for chunk in all_chunks_data)}\n\n")
        
        f.write("## Scraped URLs:\n\n")
        for page in scraped_data:
            f.write(f"- [{page['title']}]({page['url']}) ({page['word_count']} words)\n")
    
    print(f"✅ Saved {len(all_chunks_data)} chunks to {base_output_dir}")
    print(f"   - chunks.json: {chunks_json_file}")
    print(f"   - Individual chunks: {chunks_dir}/chunk_*.md")
    print(f"   - Original data: {original_data_file}")
    print(f"   - Summary: {summary_file}")
    
    return len(all_chunks_data)

def load_urls_from_file(file_path: str) -> List[str]:
    """
    Load URLs from a text file (one URL per line)
    
    Args:
        file_path: Path to file containing URLs
        
    Returns:
        List of URLs
    """
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):  # Skip empty lines and comments
                    urls.append(url)
        print(f"📄 Loaded {len(urls)} URLs from {file_path}")
        return urls
    except Exception as e:
        print(f"❌ Error loading URLs from file: {e}")
        return []

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description="Web Scraper for RAG System")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-u", "--urls", nargs='+', help="List of URLs to scrape")
    input_group.add_argument("-f", "--file", type=str, help="File containing URLs (one per line)")
    
    # Output options
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory for scraped content")
    parser.add_argument("-n", "--name", type=str, default="web_content", help="Name for the content source")
    
    # Scraping options
    parser.add_argument("-d", "--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("-r", "--retries", type=int, default=3, help="Maximum number of retries")
    
    args = parser.parse_args()
    
    # Get URLs
    if args.urls:
        urls = args.urls
    else:
        urls = load_urls_from_file(args.file)
    
    if not urls:
        print("❌ No URLs to process")
        return
    
    # Validate URLs
    valid_urls = []
    for url in urls:
        if url.startswith(('http://', 'https://')):
            valid_urls.append(url)
        else:
            print(f"⚠️  Skipping invalid URL: {url}")
    
    if not valid_urls:
        print("❌ No valid URLs found")
        return
    
    print(f"🌐 Will scrape {len(valid_urls)} valid URLs")
    
    # Initialize scraper
    scraper = WebScraper(delay=args.delay, max_retries=args.retries)
    
    try:
        # Scrape content
        scraped_data = scraper.scrape_urls(valid_urls)
        
        if not scraped_data:
            print("❌ No content was successfully scraped")
            return
        
        # Save content
        chunks_count = save_scraped_content(scraped_data, args.output, args.name)
        
        # Final summary
        print(f"\n{'='*60}")
        print("📋 SCRAPING COMPLETE - SUMMARY")
        print(f"{'='*60}")
        print(f"🌐 URLs processed: {len(scraped_data)}/{len(valid_urls)}")
        print(f"📊 Total chunks created: {chunks_count}")
        print(f"📁 Output directory: {args.output}/{args.name}")
        
        if scraped_data:
            total_words = sum(item['word_count'] for item in scraped_data)
            print(f"📝 Total words scraped: {total_words:,}")
            print(f"🎯 Average words per page: {total_words // len(scraped_data):,}")
            print("\n✅ Web content is now ready for RAG!")
            print(f"📁 Check output in: {args.output}/{args.name}/")
        
    except KeyboardInterrupt:
        print("\n🛑 Scraping interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 