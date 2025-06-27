"""
Pydantic schemas for API requests and responses.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class ToneType(str, Enum):
    """Supported tone types for ad rewriting."""
    PROFESSIONAL = "professional"
    FUN = "fun"
    CASUAL = "casual"
    URGENT = "urgent"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"


class PlatformType(str, Enum):
    """Supported platform types for ad optimization."""
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    GOOGLE = "google"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    TIKTOK = "tiktok"


class AdPerformanceData(BaseModel):
    """Ad performance data structure."""
    campaign_name: str = Field(..., description="Name of the ad campaign")
    impressions: int = Field(..., description="Number of impressions")
    clicks: int = Field(..., description="Number of clicks")
    conversions: int = Field(..., description="Number of conversions")
    spend: float = Field(..., description="Amount spent on the campaign")
    ctr: Optional[float] = Field(None, description="Click-through rate")
    cpa: Optional[float] = Field(None, description="Cost per acquisition")
    roas: Optional[float] = Field(None, description="Return on ad spend")


class AdPerformanceRequest(BaseModel):
    """Request model for ad performance analysis."""
    ad_data: List[AdPerformanceData] = Field(..., description="List of ad performance data")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")


class AdPerformanceResponse(BaseModel):
    """Response model for ad performance analysis."""
    summary: Dict[str, Any] = Field(..., description="Performance summary")
    insights: List[str] = Field(..., description="Key insights from the analysis")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    top_performers: List[str] = Field(..., description="Top performing campaigns")
    underperformers: List[str] = Field(..., description="Underperforming campaigns")
    metrics: Dict[str, float] = Field(..., description="Calculated metrics")


class BlogSearchRequest(BaseModel):
    """Request model for marketing blog search."""
    query: str = Field(..., description="Search query for marketing blogs")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    filter_category: Optional[str] = Field(None, description="Filter by category")
    filter_topic: Optional[str] = Field(None, description="Filter by topic")


class BlogSearchResult(BaseModel):
    """Individual blog search result."""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    relevance_score: float = Field(..., description="Relevance score (1 - distance)")


class BlogSearchResponse(BaseModel):
    """Response model for marketing blog search."""
    query: str = Field(..., description="Original search query")
    results: List[BlogSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    suggested_queries: List[str] = Field(default=[], description="Suggested related queries")


class AdRewriteRequest(BaseModel):
    """Request model for ad text rewriting."""
    original_text: str = Field(..., description="Original ad text to rewrite")
    target_tone: ToneType = Field(..., description="Target tone for rewriting")
    target_platform: PlatformType = Field(..., description="Target platform for optimization")
    max_length: Optional[int] = Field(None, description="Maximum length of rewritten text")
    include_cta: bool = Field(default=True, description="Include call-to-action")
    target_audience: Optional[str] = Field(None, description="Target audience description")


class AdRewriteResponse(BaseModel):
    """Response model for ad text rewriting."""
    original_text: str = Field(..., description="Original ad text")
    rewritten_text: str = Field(..., description="Rewritten ad text")
    tone_applied: ToneType = Field(..., description="Applied tone")
    platform_optimized: PlatformType = Field(..., description="Platform optimized for")
    improvements: List[str] = Field(..., description="List of improvements made")
    platform_specific_tips: List[str] = Field(..., description="Platform-specific optimization tips")
    alternative_versions: List[str] = Field(default=[], description="Alternative versions of the rewritten text")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Timestamp of health check")
    services: Dict[str, str] = Field(..., description="Status of individual services")


class AgentInfo(BaseModel):
    """Agent information model."""
    name: str = Field(..., description="Agent name")
    endpoint: str = Field(..., description="Agent endpoint")
    description: str = Field(..., description="Agent description")


class RootResponse(BaseModel):
    """Root endpoint response model."""
    message: str = Field(..., description="Welcome message")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    agents: List[AgentInfo] = Field(..., description="Available agents")
    documentation: Dict[str, str] = Field(..., description="Documentation links") 