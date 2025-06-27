"""
FastAPI router for web crawler functionality.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import logging

from app.utils.web_crawl import WebCrawler
from app.database.chroma_client import ChromaClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web-crawler", tags=["web-crawler"])


class CrawlRequest(BaseModel):
    """Request model for crawling URLs."""
    urls: List[HttpUrl]
    delay: Optional[float] = 1.0


class BlogCrawlRequest(BaseModel):
    """Request model for crawling blog sites."""
    blog_url: HttpUrl
    max_articles: Optional[int] = 10
    delay: Optional[float] = 1.5


class CrawlResponse(BaseModel):
    """Response model for crawl operations."""
    success: bool
    message: str
    crawled_urls: int
    successful_scrapes: int
    failed_scrapes: int
    collection_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/crawl-urls", response_model=CrawlResponse)
async def crawl_urls(request: CrawlRequest):
    """
    Crawl specific URLs and add content to ChromaDB.
    
    Args:
        request: CrawlRequest with URLs to crawl
        
    Returns:
        CrawlResponse with crawling results
    """
    try:
        # Convert HttpUrl objects to strings
        urls = [str(url) for url in request.urls]
        
        # Initialize crawler
        crawler = WebCrawler(delay=request.delay)
        
        # Crawl and store
        result = await crawler.crawl_and_store(urls)
        
        return CrawlResponse(
            success=result['success'],
            message=f"Successfully crawled {result['successful_scrapes']} out of {result['crawled_urls']} URLs",
            crawled_urls=result['crawled_urls'],
            successful_scrapes=result['successful_scrapes'],
            failed_scrapes=result['failed_scrapes'],
            collection_stats=result.get('collection_stats'),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error in crawl_urls: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to crawl URLs: {str(e)}")


@router.post("/crawl-blog", response_model=CrawlResponse)
async def crawl_blog_site(request: BlogCrawlRequest):
    """
    Discover and crawl articles from a blog site.
    
    Args:
        request: BlogCrawlRequest with blog URL and options
        
    Returns:
        CrawlResponse with crawling results
    """
    try:
        # Initialize crawler
        crawler = WebCrawler(delay=request.delay)
        
        # Crawl blog site
        result = await crawler.crawl_website_blog(
            str(request.blog_url), 
            max_articles=request.max_articles
        )
        
        return CrawlResponse(
            success=result['success'],
            message=f"Successfully crawled {result.get('successful_scrapes', 0)} articles from blog site",
            crawled_urls=result.get('crawled_urls', 0),
            successful_scrapes=result.get('successful_scrapes', 0),
            failed_scrapes=result.get('failed_scrapes', 0),
            collection_stats=result.get('collection_stats'),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error in crawl_blog_site: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to crawl blog site: {str(e)}")


@router.post("/crawl-urls-background")
async def crawl_urls_background(request: CrawlRequest, background_tasks: BackgroundTasks):
    """
    Crawl URLs in the background.
    
    Args:
        request: CrawlRequest with URLs to crawl
        background_tasks: FastAPI background tasks
        
    Returns:
        Message indicating task started
    """
    try:
        # Convert HttpUrl objects to strings
        urls = [str(url) for url in request.urls]
        
        # Add crawling task to background
        background_tasks.add_task(background_crawl_task, urls, request.delay)
        
        return {
            "message": f"Started background crawling for {len(urls)} URLs",
            "urls": urls
        }
        
    except Exception as e:
        logger.error(f"Error starting background crawl: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start background crawl: {str(e)}")


@router.get("/collection-stats")
async def get_collection_stats():
    """
    Get statistics about the ChromaDB collection.
    
    Returns:
        Collection statistics
    """
    try:
        # Initialize ChromaDB client
        chroma_client = ChromaClient()
        await chroma_client.initialize()
        
        # Get stats
        stats = await chroma_client.get_collection_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection stats: {str(e)}")


@router.get("/search-content")
async def search_content(query: str, n_results: int = 5):
    """
    Search through crawled content.
    
    Args:
        query: Search query
        n_results: Number of results to return
        
    Returns:
        Search results
    """
    try:
        # Initialize ChromaDB client
        chroma_client = ChromaClient()
        await chroma_client.initialize()
        
        # Search
        results = await chroma_client.search(query, n_results=n_results)
        
        return {
            "success": True,
            "query": query,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search content: {str(e)}")


async def background_crawl_task(urls: List[str], delay: float):
    """
    Background task for crawling URLs.
    
    Args:
        urls: List of URLs to crawl
        delay: Delay between requests
    """
    try:
        logger.info(f"Starting background crawl for {len(urls)} URLs")
        
        # Initialize crawler
        crawler = WebCrawler(delay=delay)
        
        # Crawl and store
        result = await crawler.crawl_and_store(urls)
        
        logger.info(f"Background crawl completed: {result}")
        
    except Exception as e:
        logger.error(f"Error in background crawl task: {e}") 