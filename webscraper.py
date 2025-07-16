"""
Enhanced Web Scraper for RAG System Integration
Scrapes websites recursively and processes content for RAG system
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
from typing import List, Dict, Optional, Set
import hashlib
from datetime import datetime
from collections import deque

# Load environment variables
load_dotenv(override=True)

class RecursiveWebScraper:
    def __init__(self, delay: float = 1.0, max_retries: int = 3, max_depth: int = 3, same_domain: bool = True):
        """
        Initialize the recursive web scraper
        
        Args:
            delay: Delay between requests in seconds
            max_retries: Maximum number of retries for failed requests
            max_depth: Maximum depth for recursive crawling
            same_domain: Whether to only crawl links from the same domain
        """
        self.delay = delay
        self.max_retries = max_retries
        self.max_depth = max_depth
        self.same_domain = same_domain
        self.session = requests.Session()
        
        # Set a respectful user agent
        self.session.headers.update({
            'User-Agent': 'RAG-Bot-WebScraper/2.0 (Educational/Research Purpose)'
        })
        
        # Track processed URLs and domains
        self.processed_urls: Set[str] = set()
        self.allowed_domains: Set[str] = set()
        
        # File extensions to avoid
        self.skip_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.webp',
            '.mp4', '.avi', '.mov', '.wmv', '.mp3', '.wav', '.zip',
            '.rar', '.tar', '.gz', '.exe', '.dmg', '.css', '.js'
        }
        
        # URL patterns to skip
        self.skip_patterns = [
            r'mailto:', r'tel:', r'javascript:', r'#', r'void\(0\)',
            r'\.css$', r'\.js$', r'\.xml$', r'\.json$', r'/feed/',
            r'/rss/', r'/sitemap', r'/robots\.txt', r'/favicon'
        ]
        
    def should_skip_url(self, url: str) -> bool:
        """
        Check if URL should be skipped based on extension or pattern
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be skipped, False otherwise
        """
        url_lower = url.lower()
        
        # Check file extensions
        for ext in self.skip_extensions:
            if url_lower.endswith(ext):
                return True
        
        # Check patterns
        for pattern in self.skip_patterns:
            if re.search(pattern, url_lower):
                return True
        
        return False
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing fragments and query parameters
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        parsed = urlparse(url)
        # Remove fragment and some query parameters
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,  # Keep query parameters for now
            ''  # Remove fragment
        ))
        return normalized
    
    def extract_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """
        Extract all valid links from a page
        
        Args:
            url: Current page URL (for resolving relative links)
            soup: BeautifulSoup object of the page
            
        Returns:
            List of valid links found on the page
        """
        links = []
        base_domain = urlparse(url).netloc
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
            
            # Resolve relative URLs
            absolute_url = urljoin(url, href)
            normalized_url = self.normalize_url(absolute_url)
            
            # Skip if already processed
            if normalized_url in self.processed_urls:
                continue
            
            # Skip if should be skipped
            if self.should_skip_url(normalized_url):
                continue
            
            # Domain filtering
            if self.same_domain:
                link_domain = urlparse(normalized_url).netloc
                if link_domain != base_domain:
                    continue
            
            # Check if domain is allowed
            if self.allowed_domains:
                link_domain = urlparse(normalized_url).netloc
                if link_domain not in self.allowed_domains:
                    continue
            
            links.append(normalized_url)
        
        return links
    
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
    
    def extract_content(self, url: str, depth: int = 0) -> Optional[Dict]:
        """
        Extract content from a single URL
        
        Args:
            url: URL to scrape
            depth: Current crawling depth
            
        Returns:
            Dictionary with extracted content or None if failed
        """
        # Check if already processed
        if url in self.processed_urls:
            return None
        
        # Check robots.txt
        if not self.check_robots_txt(url):
            print(f"🚫 Robots.txt disallows scraping: {url}")
            return None
        
        print(f"🌐 Scraping: {url} (depth: {depth})")
        
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
                
                # Extract links if we're not at max depth
                links = []
                if depth < self.max_depth:
                    links = self.extract_links(url, soup)
                
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
                    'domain': urlparse(url).netloc,
                    'depth': depth,
                    'links_found': links
                }
                
                print(f"✅ Successfully scraped: {title} ({len(content_text)} chars, {len(links)} links)")
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
    
    def scrape_recursive(self, start_urls: List[str]) -> List[Dict]:
        """
        Scrape URLs recursively using breadth-first search
        
        Args:
            start_urls: List of starting URLs
            
        Returns:
            List of scraped content dictionaries
        """
        results = []
        
        # Set up allowed domains from start URLs
        if self.same_domain:
            for url in start_urls:
                domain = urlparse(url).netloc
                self.allowed_domains.add(domain)
        
        # Use a queue for breadth-first traversal
        # Queue items: (url, depth)
        url_queue = deque([(url, 0) for url in start_urls])
        
        print(f"🚀 Starting recursive scraping from {len(start_urls)} URLs...")
        print(f"📏 Max depth: {self.max_depth}")
        print(f"🌐 Same domain only: {self.same_domain}")
        print(f"📋 Allowed domains: {', '.join(self.allowed_domains)}")
        
        while url_queue:
            current_url, depth = url_queue.popleft()
            
            # Skip if already processed
            if current_url in self.processed_urls:
                continue
            
            # Skip if depth exceeds limit
            if depth > self.max_depth:
                continue
            
            print(f"\n📍 Queue size: {len(url_queue)}, Processed: {len(self.processed_urls)}")
            
            # Scrape current URL
            content = self.extract_content(current_url, depth)
            if content:
                results.append(content)
                
                # Add found links to queue for next depth level
                if depth < self.max_depth and content.get('links_found'):
                    for link in content['links_found']:
                        if link not in self.processed_urls:
                            url_queue.append((link, depth + 1))
        
        print(f"\n✅ Recursive scraping complete!")
        print(f"📊 Total pages scraped: {len(results)}")
        print(f"🔗 Total URLs processed: {len(self.processed_urls)}")
        
        return results
    
    def scrape_urls(self, urls: List[str]) -> List[Dict]:
        """
        Scrape multiple URLs (non-recursive, for backward compatibility)
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped content dictionaries
        """
        results = []
        
        print(f"🚀 Starting to scrape {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            print(f"\n📍 Progress: {i}/{len(urls)}")
            
            content = self.extract_content(url, 0)
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
        depth = page_data.get('depth', 0)
        links_found = page_data.get('links_found', [])
        
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
**Crawl Depth:** {depth}
**Links Found:** {len(links_found)}

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
                'depth': depth,
                'links_found_count': len(links_found),
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
        f.write(f"**Total Word Count:** {sum(chunk['word_count'] for chunk in all_chunks_data)}\n")
        
        # Additional stats for recursive crawling
        if any(page.get('depth', 0) > 0 for page in scraped_data):
            depths = [page.get('depth', 0) for page in scraped_data]
            f.write(f"**Max Depth Reached:** {max(depths)}\n")
            f.write(f"**Total Links Found:** {sum(len(page.get('links_found', [])) for page in scraped_data)}\n")
            
            # Pages by depth
            depth_counts = {}
            for depth in depths:
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
            
            f.write(f"\n## Pages by Depth:\n\n")
            for depth in sorted(depth_counts.keys()):
                f.write(f"- Depth {depth}: {depth_counts[depth]} pages\n")
        
        f.write("\n## Scraped URLs:\n\n")
        for page in scraped_data:
            depth_info = f" (depth {page.get('depth', 0)})" if page.get('depth', 0) > 0 else ""
            f.write(f"- [{page['title']}]({page['url']}) ({page['word_count']} words{depth_info})\n")
    
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
    parser = argparse.ArgumentParser(description="Enhanced Web Scraper for RAG System with Recursive Crawling")
    
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
    
    # Recursive crawling options
    parser.add_argument("-D", "--depth", type=int, default=3, help="Maximum depth for recursive crawling")
    parser.add_argument("-s", "--same-domain", action="store_true", help="Only crawl links from the same domain")
    parser.add_argument("-R", "--recursive", action="store_true", help="Enable recursive crawling")
    
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
    scraper = RecursiveWebScraper(
        delay=args.delay,
        max_retries=args.retries,
        max_depth=args.depth,
        same_domain=args.same_domain
    )
    
    try:
        # Choose scraping method
        if args.recursive:
            print("🔄 Using recursive crawling mode")
            scraped_data = scraper.scrape_recursive(valid_urls)
        else:
            print("📄 Using single-page mode")
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
            
            # Additional stats for recursive crawling
            if args.recursive:
                depths = [item['depth'] for item in scraped_data]
                print(f"📏 Max depth reached: {max(depths)}")
                print(f"🔗 Total unique URLs found: {len(scraper.processed_urls)}")
                
                # Pages by depth
                depth_counts = {}
                for depth in depths:
                    depth_counts[depth] = depth_counts.get(depth, 0) + 1
                
                print("📊 Pages by depth:")
                for depth in sorted(depth_counts.keys()):
                    print(f"   Depth {depth}: {depth_counts[depth]} pages")
            
            print("\n✅ Web content is now ready for RAG!")
            print(f"📁 Check output in: {args.output}/{args.name}/")
        
    except KeyboardInterrupt:
        print("\n🛑 Scraping interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 