"""
FastAPI router for Ad Performance Analyzer agent.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AdPerformanceRequest, AdPerformanceResponse, ErrorResponse
)
from app.agents.ad_performance_agent import AdPerformanceAgent

logger = logging.getLogger(__name__)

router = APIRouter()

ad_performance_agent = AdPerformanceAgent()


@router.post(
    "/analyze-ad-performance",
    response_model=AdPerformanceResponse,
    summary="Analyze Ad Performance",
    description="Analyze Meta/Google ad performance data and provide insights and recommendations",
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"description": "Invalid request data"},
        500: {"description": "Internal server error"}
    }
)
async def analyze_ad_performance(request: AdPerformanceRequest) -> AdPerformanceResponse:
    """
    Analyze ad performance data and generate insights.
    
    This endpoint accepts ad performance data from Meta (Facebook/Instagram) or Google Ads
    and returns comprehensive analysis including:
    - Performance insights and trends
    - Improvement recommendations
    - Top and underperforming campaigns
    - Key metrics calculations
    
    The analysis uses AI to identify patterns and suggest actionable improvements.
    """
    try:
        logger.info(f"Received ad performance analysis request for {len(request.ad_data)} campaigns")
        
        if not request.ad_data:
            raise HTTPException(
                status_code=400,
                detail="No ad performance data provided"
            )
        
        if len(request.ad_data) > 100:
            raise HTTPException(
                status_code=400,
                detail="Too many campaigns provided. Maximum 100 campaigns allowed."
            )
        
        for i, ad_data in enumerate(request.ad_data):
            if ad_data.impressions < 0 or ad_data.clicks < 0 or ad_data.conversions < 0 or ad_data.spend < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Campaign '{ad_data.campaign_name}' has negative values which are not allowed"
                )
            
            if ad_data.clicks > ad_data.impressions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Campaign '{ad_data.campaign_name}' has more clicks than impressions"
                )
        
        result = await ad_performance_agent.analyze_performance(
            ad_data=request.ad_data,
            analysis_type=request.analysis_type
        )
        
        logger.info("Ad performance analysis completed successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze ad performance: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while analyzing ad performance"
        )


@router.get(
    "/ad-performance/sample-data",
    summary="Get Sample Ad Performance Data",
    description="Get sample ad performance data for testing the analysis endpoint",
    response_model=AdPerformanceRequest
)
async def get_sample_ad_performance_data() -> AdPerformanceRequest:
    """
    Get sample ad performance data for testing purposes.
    
    Returns sample campaign data that can be used to test the ad performance analysis endpoint.
    """
    try:
        from app.models.schemas import AdPerformanceData
        
        sample_data = [
            AdPerformanceData(
                campaign_name="Summer Sale 2024 - Facebook",
                impressions=50000,
                clicks=2500,
                conversions=125,
                spend=750.00,
                ctr=5.0,
                cpa=6.00,
                roas=3.2
            ),
            AdPerformanceData(
                campaign_name="Back to School - Instagram",
                impressions=35000,
                clicks=1400,
                conversions=70,
                spend=420.00,
                ctr=4.0,
                cpa=6.00,
                roas=2.8
            ),
            AdPerformanceData(
                campaign_name="Holiday Promo - Google Ads",
                impressions=25000,
                clicks=3750,
                conversions=300,
                spend=1200.00,
                ctr=15.0,
                cpa=4.00,
                roas=4.5
            ),
            AdPerformanceData(
                campaign_name="Brand Awareness - LinkedIn",
                impressions=15000,
                clicks=450,
                conversions=18,
                spend=300.00,
                ctr=3.0,
                cpa=16.67,
                roas=1.2
            ),
            AdPerformanceData(
                campaign_name="Retargeting Campaign - Facebook",
                impressions=20000,
                clicks=2000,
                conversions=200,
                spend=600.00,
                ctr=10.0,
                cpa=3.00,
                roas=5.0
            )
        ]
        
        return AdPerformanceRequest(
            ad_data=sample_data,
            analysis_type="comprehensive"
        )
        
    except Exception as e:
        logger.error(f"Failed to generate sample data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate sample data"
        )


@router.get(
    "/ad-performance/metrics-info",
    summary="Get Ad Performance Metrics Information",
    description="Get information about the metrics used in ad performance analysis"
)
async def get_ad_performance_metrics_info():
    """
    Get information about ad performance metrics.
    
    Returns descriptions of the metrics used in the analysis and their calculations.
    """
    return {
        "metrics": {
            "CTR (Click-Through Rate)": {
                "description": "Percentage of impressions that resulted in clicks",
                "calculation": "(Clicks / Impressions) × 100",
                "good_benchmark": "> 2% for display, > 3% for search"
            },
            "CPA (Cost Per Acquisition)": {
                "description": "Average cost to acquire one conversion",
                "calculation": "Total Spend / Total Conversions",
                "note": "Lower is better"
            },
            "Conversion Rate": {
                "description": "Percentage of clicks that resulted in conversions",
                "calculation": "(Conversions / Clicks) × 100",
                "good_benchmark": "> 2-5% depending on industry"
            },
            "CPM (Cost Per Mille)": {
                "description": "Cost per 1,000 impressions",
                "calculation": "(Total Spend / Impressions) × 1000",
                "note": "Varies by platform and targeting"
            },
            "CPC (Cost Per Click)": {
                "description": "Average cost per click",
                "calculation": "Total Spend / Total Clicks",
                "note": "Lower is generally better"
            },
            "ROAS (Return on Ad Spend)": {
                "description": "Revenue generated per dollar spent on ads",
                "calculation": "Revenue / Ad Spend",
                "good_benchmark": "> 4:1 for most businesses"
            }
        },
        "analysis_types": {
            "comprehensive": "Full analysis including insights, recommendations, and performance tiers",
            "quick": "Basic metrics calculation and top-level insights",
            "comparison": "Focus on comparing campaigns against each other"
        }
    } 