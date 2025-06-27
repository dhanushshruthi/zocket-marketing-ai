"""
ChromaDB client for vector database operations.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np

from app.utils.config import get_settings

logger = logging.getLogger(__name__)


class ChromaClient:
    """ChromaDB client for managing vector embeddings and similarity search."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.settings = get_settings()
        
    async def initialize(self):
        """Initialize ChromaDB client and embedding model."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.settings.chroma_db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.settings.chroma_db_path
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"description": "Marketing knowledge base"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.settings.chroma_collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def initialize_sample_data(self):
        """Initialize sample marketing blog data."""
        try:
            # Check if collection already has data
            count = self.collection.count()
            if count > 0:
                logger.info(f"Collection already has {count} documents")
                return
            
            # Sample marketing blog data
            sample_data = [
                {
                    "id": "blog_1",
                    "content": "Best practices for summer sale campaigns include creating urgency with limited-time offers, using bright and energetic visuals, targeting vacation-minded consumers, and leveraging social media for maximum reach. Summer campaigns should focus on seasonal products and experiences.",
                    "metadata": {
                        "source": "marketing_blog",
                        "topic": "summer_sales",
                        "category": "campaign_strategy"
                    }
                },
                {
                    "id": "blog_2", 
                    "content": "Effective ad copy for Facebook campaigns should be concise, include a clear call-to-action, use emotional triggers, test different headlines, and incorporate social proof. A/B testing different copy variations is crucial for optimization.",
                    "metadata": {
                        "source": "marketing_blog",
                        "topic": "facebook_ads",
                        "category": "ad_copy"
                    }
                },
                {
                    "id": "blog_3",
                    "content": "Google Ads performance can be improved by focusing on keyword relevance, improving quality scores, optimizing landing pages, using negative keywords, and implementing proper bidding strategies. Regular performance monitoring is essential.",
                    "metadata": {
                        "source": "marketing_blog",
                        "topic": "google_ads",
                        "category": "performance_optimization"
                    }
                },
                {
                    "id": "blog_4",
                    "content": "Professional tone in marketing communications builds trust and credibility. Use formal language, avoid slang, focus on benefits rather than features, and maintain consistency across all touchpoints. Professional tone works well for B2B marketing.",
                    "metadata": {
                        "source": "marketing_blog",
                        "topic": "tone_strategy",
                        "category": "professional_tone"
                    }
                },
                {
                    "id": "blog_5",
                    "content": "Fun and casual tone in advertising appeals to younger demographics and lifestyle brands. Use conversational language, emojis, humor, and trendy references. This tone works well for social media and consumer products.",
                    "metadata": {
                        "source": "marketing_blog",
                        "topic": "tone_strategy",
                        "category": "casual_tone"
                    }
                },
                {
                    "id": "blog_6",
                    "content": "Cross-platform advertising requires adapting content for each platform's unique characteristics. Instagram focuses on visuals, LinkedIn on professional content, TikTok on entertainment, and Facebook on community engagement.",
                    "metadata": {
                        "source": "marketing_blog",
                        "topic": "platform_optimization",
                        "category": "multi_platform"
                    }
                },
                {
                    "id": "blog_7",
                    "content": "Ad performance metrics to track include click-through rate (CTR), conversion rate, cost per acquisition (CPA), return on ad spend (ROAS), and quality score. These metrics help optimize campaign effectiveness.",
                    "metadata": {
                        "source": "marketing_blog",
                        "topic": "performance_metrics",
                        "category": "analytics"
                    }
                },
                {
                    "id": "blog_8",
                    "content": "Creative improvement strategies for underperforming ads include testing new visuals, updating copy, changing call-to-action buttons, adjusting targeting parameters, and analyzing competitor strategies.",
                    "metadata": {
                        "source": "marketing_blog",
                        "topic": "creative_optimization",
                        "category": "improvement_strategies"
                    }
                }
            ]
            
            # Add documents to collection
            await self.add_documents(sample_data)
            logger.info(f"Added {len(sample_data)} sample documents to collection")
            
        except Exception as e:
            logger.error(f"Failed to initialize sample data: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the collection."""
        try:
            contents = [doc["content"] for doc in documents]
            ids = [doc["id"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(contents).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    async def search(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.settings.chroma_collection_name,
                "document_count": count,
                "embedding_model": "all-MiniLM-L6-v2"
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
    
    async def close(self):
        """Close the ChromaDB client."""
        try:
            if self.client:
                # ChromaDB doesn't require explicit closing
                logger.info("ChromaDB client closed")
        except Exception as e:
            logger.error(f"Failed to close ChromaDB client: {e}") 