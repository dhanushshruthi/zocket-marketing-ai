"""
Marketing AI Agents FastAPI Application

This application provides three AI agents for marketing research tasks:
1. Ad Performance Analyzer - Reviews Meta/Google ad performance CSVs
2. Marketing Blog Search Agent - Multi-step agent using vector database
3. Ad Text Rewriter - Rewrites ad text with different tones and platform optimization
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn

from app.routers import ad_performance, blog_search, ad_rewriter, web_crawler
from app.database.chroma_client import ChromaClient
from app.utils.config import get_settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for database clients
chroma_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global chroma_client
    
    # Startup
    logger.info("Starting Marketing AI Agents application...")
    
    # Initialize ChromaDB client
    try:
        chroma_client = ChromaClient()
        await chroma_client.initialize()
        app.state.chroma_client = chroma_client
        logger.info("ChromaDB client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        raise
    
    # Initialize sample marketing blog data
    try:
        await chroma_client.initialize_sample_data()
        logger.info("Sample marketing data initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize sample data: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Marketing AI Agents application...")
    if chroma_client:
        await chroma_client.close()

# Create FastAPI application
app = FastAPI(
    title="Marketing AI Agents",
    description="Lightweight AI agents for marketing research tasks using Azure OpenAI and ChromaDB",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ad_performance.router, prefix="/api/v1", tags=["Ad Performance"])
app.include_router(blog_search.router, prefix="/api/v1", tags=["Blog Search"])
app.include_router(ad_rewriter.router, prefix="/api/v1", tags=["Ad Rewriter"])
app.include_router(web_crawler.router, prefix="/api/v1", tags=["Web Crawler"])

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Marketing AI Agents API",
        "version": "1.0.0",
        "description": "Lightweight AI agents for marketing research tasks",
        "agents": [
            {
                "name": "Ad Performance Analyzer",
                "endpoint": "/api/v1/analyze-ad-performance",
                "description": "Reviews Meta/Google ad performance CSVs and provides insights"
            },
            {
                "name": "Marketing Blog Search Agent", 
                "endpoint": "/api/v1/search-marketing-blogs",
                "description": "Multi-step agent using vector database to search marketing blogs"
            },
            {
                "name": "Ad Text Rewriter",
                "endpoint": "/api/v1/rewrite-ad-text",
                "description": "Rewrites ad text with different tones and platform optimization"
            },
            {
                "name": "Web Crawler",
                "endpoint": "/api/v1/web-crawler",
                "description": "Scrapes blog content from URLs and adds to ChromaDB knowledge base"
            }
        ],
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    
    health_status = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",  # In production, use actual timestamp
        "services": {
            "azure_openai": "connected" if settings.azure_openai_api_key else "not_configured",
            "chroma_db": "connected" if chroma_client else "not_connected",
        }
    }
    
    return health_status

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )
for route in app.routes:
    print(route.path)


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    ) 