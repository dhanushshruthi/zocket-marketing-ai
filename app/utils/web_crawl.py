"""
Web crawler for scraping blog content and adding to ChromaDB.
"""

import asyncio
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import time

from app.database.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


class WebCrawler:
    """Web crawler for scraping blog content and storing in ChromaDB."""
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize web crawler.
        
        Args:
            delay: Delay between requests in seconds (default: 1.0)
        """
        self.chroma_client = ChromaClient()
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def generate_document_id(self, url: str, title: str = "") -> str:
        """Generate a unique document ID from URL and title."""
        content = f"{url}_{title}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        return text
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract article content from HTML soup.
        
        Args:
            soup: BeautifulSoup object of the webpage
            url: URL of the webpage
            
        Returns:
            Dictionary with title, content, and metadata
        """
        # Try to extract title
        title = ""
        title_selectors = ['h1', 'title', '.post-title', '.entry-title', '.article-title']
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text().strip()
                break
        
        # Try to extract main content
        content = ""
        content_selectors = [
            'article', '.post-content', '.entry-content', '.article-content',
            '.content', 'main', '.post', '.blog-post'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove script, style, and nav elements
                for tag in content_elem.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    tag.decompose()
                content = content_elem.get_text()
                break
        
        # If no specific content area found, use body but filter out navigation
        if not content:
            body = soup.find('body')
            if body:
                # Remove common navigation and footer elements
                for tag in body.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                    tag.decompose()
                content = body.get_text()
        
        # Clean the content
        content = self.clean_text(content)
        title = self.clean_text(title)
        
        # Extract meta description if available
        description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content', '').strip()
        
        # Extract publish date if available
        publish_date = None
        date_selectors = [
            'time[datetime]', '.publish-date', '.post-date', '.entry-date'
        ]
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                if date_elem.name == 'time':
                    publish_date = date_elem.get('datetime')
                else:
                    publish_date = date_elem.get_text().strip()
                break
        
        return {
            'title': title,
            'content': content,
            'description': description,
            'publish_date': publish_date,
            'url': url,
            'domain': urlparse(url).netloc
        }
    
    def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single webpage.
        
        Args:
            url: URL to scrape
            
        Returns:
            Document dictionary or None if failed
        """
        try:
            logger.info(f"Scraping: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            article_data = self.extract_article_content(soup, url)
            
            # Skip if content is too short (likely not a proper article)
            if len(article_data['content']) < 200:
                logger.warning(f"Content too short for {url}, skipping")
                return None
            
            # Create document for ChromaDB
            doc_id = self.generate_document_id(url, article_data['title'])
            
            document = {
                'id': doc_id,
                'content': article_data['content'][:4000],  # Limit content length
                'metadata': {
                    'source': 'web_crawl',
                    'url': url,
                    'title': article_data['title'],
                    'domain': article_data['domain'],
                    'description': article_data['description'],
                    'publish_date': article_data['publish_date'],
                    'scraped_at': datetime.now().isoformat(),
                    'category': 'blog_article'
                }
            }
            
            return document
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def discover_blog_urls(self, base_url: str, max_pages: int = 10) -> List[str]:
        """
        Discover blog URLs from a website's blog/news section.
        
        Args:
            base_url: Base URL of the website
            max_pages: Maximum number of pages to discover
            
        Returns:
            List of discovered URLs
        """
        urls = []
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links that might be blog posts
            link_selectors = [
                'a[href*="/blog/"]', 'a[href*="/post/"]', 'a[href*="/article/"]',
                'a[href*="/news/"]', '.post-link a', '.blog-link a',
                '.article-link a', 'article a', '.entry-title a'
            ]
            
            found_links = set()
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links[:max_pages]:  # Limit number of links
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        found_links.add(full_url)
            
            urls = list(found_links)[:max_pages]
            logger.info(f"Discovered {len(urls)} URLs from {base_url}")
            
        except Exception as e:
            logger.error(f"Error discovering URLs from {base_url}: {e}")
        
        return urls
    
    async def crawl_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Crawl multiple URLs and return documents.
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        for i, url in enumerate(urls):
            try:
                document = self.scrape_page(url)
                if document:
                    documents.append(document)
                    logger.info(f"Successfully scraped {url}")
                else:
                    logger.warning(f"Failed to scrape {url}")
                
                # Add delay between requests
                if i < len(urls) - 1:  # Don't delay after last request
                    time.sleep(self.delay)
                    
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue
        
        return documents
    
    async def crawl_and_store(self, urls: List[str]) -> Dict[str, Any]:
        """
        Crawl URLs and store in ChromaDB.
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            Summary of the crawling process
        """
        try:
            # Initialize ChromaDB
            await self.chroma_client.initialize()
            
            # Crawl URLs
            documents = await self.crawl_urls(urls)
            
            if documents:
                # Add documents to ChromaDB
                await self.chroma_client.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to ChromaDB")
            
            # Get collection stats
            stats = await self.chroma_client.get_collection_stats()
            
            return {
                'success': True,
                'crawled_urls': len(urls),
                'successful_scrapes': len(documents),
                'failed_scrapes': len(urls) - len(documents),
                'collection_stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error in crawl_and_store: {e}")
            return {
                'success': False,
                'error': str(e),
                'crawled_urls': len(urls),
                'successful_scrapes': 0,
                'failed_scrapes': len(urls)
            }
    
    async def crawl_website_blog(self, base_url: str, max_articles: int = 20) -> Dict[str, Any]:
        """
        Crawl a website's blog section.
        
        Args:
            base_url: Base URL of the website's blog
            max_articles: Maximum number of articles to crawl
            
        Returns:
            Summary of the crawling process
        """
        try:
            # Discover blog URLs
            discovered_urls = self.discover_blog_urls(base_url, max_articles)
            
            if not discovered_urls:
                # If no URLs discovered, try the base URL itself
                discovered_urls = [base_url]
            
            # Crawl and store
            result = await self.crawl_and_store(discovered_urls)
            result['base_url'] = base_url
            result['discovered_urls'] = len(discovered_urls)
            
            return result
            
        except Exception as e:
            logger.error(f"Error crawling website blog {base_url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'base_url': base_url
            }


async def main():
    """Example usage of the web crawler."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example marketing blog URLs to crawl
    example_urls = [
        "https://blog.hubspot.com/marketing",
        "https://blog.hootsuite.com/",
        "https://contentmarketinginstitute.com/blog/",
        "https://blog.marketo.com/",
        "https://blog.mailchimp.com/"
    ]
    
    # You can also specify individual article URLs
    individual_articles = [
        "https://blog.hubspot.com/marketing/google-ads-optimization",
        "https://blog.hootsuite.com/social-media-advertising/",
    ]
    
    crawler = WebCrawler(delay=1.5)  # 1.5 second delay between requests
    
    # Crawl individual URLs
    if individual_articles:
        print("Crawling individual articles...")
        result = await crawler.crawl_and_store(individual_articles)
        print(f"Results: {result}")
    
    # Crawl website blogs
    for blog_url in example_urls[:2]:  # Limit to first 2 for testing
        print(f"\nCrawling blog: {blog_url}")
        result = await crawler.crawl_website_blog(blog_url, max_articles=5)
        print(f"Results: {result}")


if __name__ == "__main__":
    asyncio.run(main()) 