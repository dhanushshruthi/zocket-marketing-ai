"""
Example script demonstrating how to use the web crawler to scrape blog content.
"""

import asyncio
import logging
import sys
import os

# Add the app directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.utils.web_crawl import WebCrawler


async def crawl_specific_articles():
    """Example: Crawl specific article URLs."""
    print("=== Crawling Specific Articles ===")
    
    # List of specific marketing blog articles
    article_urls = [
        "https://blog.hubspot.com/marketing/how-to-use-facebook-ads-manager",
        "https://blog.hootsuite.com/social-media-advertising/",
        "https://blog.mailchimp.com/email-marketing-guide/",
        "https://contentmarketinginstitute.com/articles/content-marketing-strategy-guide/"
    ]
    
    crawler = WebCrawler(delay=1.0)
    result = await crawler.crawl_and_store(article_urls)
    
    print(f"✅ Crawled {result['successful_scrapes']} out of {result['crawled_urls']} articles")
    print(f"📊 Collection now has {result['collection_stats']['document_count']} total documents")
    return result


async def crawl_blog_sites():
    """Example: Discover and crawl blog sites automatically."""
    print("\n=== Auto-Discovering Blog Articles ===")
    
    # Blog sites to crawl (the crawler will discover individual articles)
    blog_sites = [
        "https://blog.hubspot.com/marketing",
        "https://blog.hootsuite.com/",
        "https://contentmarketinginstitute.com/blog/"
    ]
    
    crawler = WebCrawler(delay=1.5)
    
    results = []
    for blog_url in blog_sites:
        print(f"\n🔍 Crawling blog site: {blog_url}")
        result = await crawler.crawl_website_blog(blog_url, max_articles=3)
        results.append(result)
        
        if result['success']:
            print(f"✅ Successfully crawled {result['successful_scrapes']} articles")
            print(f"🔗 Discovered {result['discovered_urls']} URLs")
        else:
            print(f"❌ Failed to crawl {blog_url}: {result.get('error', 'Unknown error')}")
    
    return results


async def search_crawled_content():
    """Example: Search through crawled content."""
    print("\n=== Searching Crawled Content ===")
    
    from app.database.chroma_client import ChromaClient
    
    # Initialize ChromaDB client
    chroma_client = ChromaClient()
    await chroma_client.initialize()
    
    # Search for marketing-related content
    search_queries = [
        "Facebook advertising best practices",
        "Google Ads optimization",
        "content marketing strategy",
        "social media marketing tips"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = await chroma_client.search(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata'].get('title', 'No title')}")
            print(f"     📍 {result['metadata'].get('url', 'No URL')}")
            print(f"     📊 Similarity: {1 - result['distance']:.3f}")
            print(f"     💬 Preview: {result['content'][:200]}...")
            print()


async def main():
    """Main function to run all examples."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Web Crawler Examples")
    print("=" * 50)
    
    try:
        # Example 1: Crawl specific articles
        await crawl_specific_articles()
        
        # Example 2: Auto-discover and crawl blog sites
        await crawl_blog_sites()
        
        # Example 3: Search through crawled content
        await search_crawled_content()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 