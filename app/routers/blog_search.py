"""
FastAPI router for Marketing Blog Search agent.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request

from app.models.schemas import (
    BlogSearchRequest, BlogSearchResponse, ErrorResponse
)
from app.agents.blog_search_agent import BlogSearchAgent
from app.database.chroma_client import ChromaClient

logger = logging.getLogger(__name__)

router = APIRouter()


def get_blog_search_agent(request: Request) -> BlogSearchAgent:
    """Dependency to get blog search agent with ChromaDB client."""
    chroma_client = request.app.state.chroma_client
    if not chroma_client:
        raise HTTPException(
            status_code=500,
            detail="ChromaDB client not initialized"
        )
    return BlogSearchAgent(chroma_client)


@router.post(
    "/search-marketing-blogs",
    response_model=BlogSearchResponse,
    summary="Search Marketing Blogs",
    description="Multi-step agent that searches marketing blogs using vector database and provides comprehensive answers",
    responses={
        200: {"description": "Search completed successfully"},
        400: {"description": "Invalid request data"},
        500: {"description": "Internal server error"}
    }
)
async def search_marketing_blogs(
    request: BlogSearchRequest,
    agent: BlogSearchAgent = Depends(get_blog_search_agent)
) -> BlogSearchResponse:
    """
    Search marketing blogs and generate comprehensive answers.
    
    This endpoint uses Agentic RAG (Retrieval Augmented Generation) to:
    - Perform semantic search across marketing blog content
    - Expand queries for comprehensive coverage
    - Generate AI-enhanced relevance explanations
    - Provide suggested related queries
    
    The agent uses ChromaDB for vector search and Azure OpenAI for query understanding
    and response enhancement.
    """
    try:
        logger.info(f"Received blog search request: '{request.query}'")
        
        # Validate input
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 3 characters long"
            )
        
        if request.max_results < 1 or request.max_results > 50:
            raise HTTPException(
                status_code=400,
                detail="max_results must be between 1 and 50"
            )
        
        # Perform search
        result = await agent.search_and_answer(request)
        
        logger.info(f"Blog search completed: {result.total_results} results in {result.search_time_ms}ms")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search marketing blogs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while searching marketing blogs"
        )


@router.get(
    "/blog-search/collection-stats",
    summary="Get Blog Collection Statistics",
    description="Get statistics about the marketing blog collection in ChromaDB"
)
async def get_blog_collection_stats(request: Request):
    """
    Get statistics about the marketing blog collection.
    
    Returns information about the number of documents, collection name, and embedding model used.
    """
    try:
        chroma_client = request.app.state.chroma_client
        if not chroma_client:
            raise HTTPException(
                status_code=500,
                detail="ChromaDB client not initialized"
            )
        
        stats = await chroma_client.get_collection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve collection statistics"
        )


@router.get(
    "/blog-search/available-topics",
    summary="Get Available Topics",
    description="Get list of available topics for filtering blog search results"
)
async def get_available_topics(request: Request):
    """
    Get available topics and categories for filtering.
    
    Returns the topics and categories available in the marketing blog collection
    that can be used for filtering search results.
    """
    try:
        # These are based on the sample data initialized in ChromaDB
        return {
            "topics": [
                "summer_sales",
                "facebook_ads",
                "google_ads",
                "tone_strategy",
                "platform_optimization",
                "performance_metrics",
                "creative_optimization"
            ],
            "categories": [
                "campaign_strategy",
                "ad_copy",
                "performance_optimization",
                "professional_tone",
                "casual_tone",
                "multi_platform",
                "analytics",
                "improvement_strategies"
            ],
            "usage_examples": {
                "filter_by_topic": "Add 'filter_topic': 'facebook_ads' to search only Facebook-related content",
                "filter_by_category": "Add 'filter_category': 'ad_copy' to search only ad copywriting content",
                "combined_filtering": "Use both topic and category filters for more specific results"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get available topics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve available topics"
        )


@router.get(
    "/blog-search/sample-queries",
    summary="Get Sample Search Queries",
    description="Get sample marketing-related queries to test the blog search functionality"
)
async def get_sample_queries():
    """
    Get sample marketing queries for testing.
    
    Returns a list of sample queries that demonstrate the blog search capabilities
    across different marketing topics and use cases.
    """
    return {
        "sample_queries": [
            {
                "query": "Best ad copy for summer sale campaigns",
                "description": "Search for summer campaign strategies and copywriting tips",
                "expected_topics": ["summer_sales", "campaign_strategy", "ad_copy"]
            },
            {
                "query": "How to improve Facebook ad performance",
                "description": "Find strategies for optimizing Facebook advertising campaigns",
                "expected_topics": ["facebook_ads", "performance_optimization"]
            },
            {
                "query": "Professional tone in marketing communications",
                "description": "Learn about using professional tone in marketing content",
                "expected_topics": ["tone_strategy", "professional_tone"]
            },
            {
                "query": "Cross-platform advertising best practices",
                "description": "Discover strategies for multi-platform advertising",
                "expected_topics": ["platform_optimization", "multi_platform"]
            },
            {
                "query": "Google Ads optimization techniques",
                "description": "Find methods to improve Google Ads campaign performance",
                "expected_topics": ["google_ads", "performance_optimization"]
            },
            {
                "query": "Creative improvement for underperforming ads",
                "description": "Get suggestions for improving ad creative elements",
                "expected_topics": ["creative_optimization", "improvement_strategies"]
            },
            {
                "query": "Fun and casual advertising tone",
                "description": "Learn about using casual and fun tones in advertising",
                "expected_topics": ["tone_strategy", "casual_tone"]
            },
            {
                "query": "Key metrics for ad performance tracking",
                "description": "Understand important metrics for measuring ad success",
                "expected_topics": ["performance_metrics", "analytics"]
            }
        ],
        "usage_tips": [
            "Use specific keywords related to your marketing challenge",
            "Include platform names (Facebook, Google, Instagram) for platform-specific advice",
            "Ask about specific metrics (CTR, CPA, ROAS) for targeted information",
            "Combine topics like 'professional tone LinkedIn ads' for more specific results"
        ]
    }


@router.post(
    "/blog-search/add-content",
    summary="Add Marketing Content",
    description="Add new marketing content to the knowledge base (Admin only)",
    responses={
        200: {"description": "Content added successfully"},
        400: {"description": "Invalid content data"},
        500: {"description": "Internal server error"}
    }
)
async def add_marketing_content(
    content_data: dict,
    request: Request
):
    """
    Add new marketing content to the ChromaDB collection.
    
    This endpoint allows adding new marketing blog content to expand the knowledge base.
    Requires proper content structure with id, content, and metadata fields.
    
    Note: In production, this should be protected with authentication and authorization.
    """
    try:
        chroma_client = request.app.state.chroma_client
        if not chroma_client:
            raise HTTPException(
                status_code=500,
                detail="ChromaDB client not initialized"
            )
        
        # Validate content data structure
        required_fields = ['id', 'content', 'metadata']
        if not all(field in content_data for field in required_fields):
            raise HTTPException(
                status_code=400,
                detail=f"Content data must include all fields: {required_fields}"
            )
        
        if len(content_data['content']) < 50:
            raise HTTPException(
                status_code=400,
                detail="Content must be at least 50 characters long"
            )
        
        # Add content to ChromaDB
        await chroma_client.add_documents([content_data])
        
        logger.info(f"Added new content with ID: {content_data['id']}")
        
        return {
            "message": "Content added successfully",
            "content_id": content_data['id'],
            "content_length": len(content_data['content'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add marketing content: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to add content to knowledge base"
        ) 