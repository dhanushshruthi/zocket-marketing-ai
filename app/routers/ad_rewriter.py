"""
FastAPI router for Ad Text Rewriter agent.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AdRewriteRequest, AdRewriteResponse, ToneType, PlatformType, ErrorResponse
)
from app.agents.ad_rewriter_agent import AdRewriterAgent

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize agent (will be created once per process)
ad_rewriter_agent = AdRewriterAgent()


@router.post(
    "/rewrite-ad-text",
    response_model=AdRewriteResponse,
    summary="Rewrite Ad Text",
    description="Rewrite ad text with different tones and optimize for specific platforms",
    responses={
        200: {"description": "Ad text rewritten successfully"},
        400: {"description": "Invalid request data"},
        500: {"description": "Internal server error"}
    }
)
async def rewrite_ad_text(request: AdRewriteRequest) -> AdRewriteResponse:
    """
    Rewrite ad text with specified tone and platform optimization.
    
    This endpoint uses AI to rewrite advertising text with:
    - Different tones (professional, fun, casual, urgent, friendly, authoritative)
    - Platform-specific optimization (Facebook, Instagram, Google, LinkedIn, Twitter, TikTok)
    - Character limit considerations
    - Call-to-action optimization
    - Alternative versions for A/B testing
    
    The agent considers platform best practices, audience expectations, and conversion optimization.
    """
    try:
        logger.info(f"Received ad rewrite request for {request.target_tone} tone and {request.target_platform} platform")
        
        # Validate input
        if not request.original_text or len(request.original_text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Original text must be at least 10 characters long"
            )
        
        if len(request.original_text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Original text is too long. Maximum 5000 characters allowed."
            )
        
        if request.max_length and request.max_length < 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum length must be at least 20 characters if specified"
            )
        
        # Perform rewriting
        result = await ad_rewriter_agent.rewrite_ad_text(request)
        
        logger.info("Ad text rewriting completed successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rewrite ad text: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while rewriting ad text"
        )


@router.get(
    "/ad-rewriter/available-options",
    summary="Get Available Options",
    description="Get available tone types and platform types for ad rewriting"
)
async def get_available_options():
    """
    Get available tone and platform options.
    
    Returns all supported tone types and platforms that can be used for ad rewriting.
    """
    return {
        "tones": [tone.value for tone in ToneType],
        "platforms": [platform.value for platform in PlatformType],
        "tone_descriptions": {
            "professional": "Formal, credible, trustworthy",
            "fun": "Playful, energetic, entertaining",
            "casual": "Relaxed, conversational, approachable",
            "urgent": "Time-sensitive, action-oriented",
            "friendly": "Warm, welcoming, supportive",
            "authoritative": "Expert, confident, knowledgeable"
        },
        "platform_characteristics": {
            "facebook": "Community-focused, diverse audience",
            "instagram": "Visual-first, younger demographics",
            "google": "Search-intent driven, conversion-focused",
            "linkedin": "Professional, B2B oriented",
            "twitter": "Real-time, news-focused",
            "tiktok": "Entertainment, Gen Z audience"
        }
    }


@router.get(
    "/ad-rewriter/sample-request",
    summary="Get Sample Request",
    description="Get a sample ad rewrite request for testing"
)
async def get_sample_request():
    """
    Get a sample ad rewrite request.
    
    Returns a complete sample request that can be used to test the ad rewriter endpoint.
    """
    return {
        "sample_request": {
            "original_text": "Our premium skincare line delivers amazing results. Made with natural ingredients and backed by science. Free shipping on orders over $50. Try it risk-free with our 30-day money-back guarantee.",
            "target_tone": "professional",
            "target_platform": "facebook",
            "max_length": 200,
            "include_cta": True,
            "target_audience": "Women aged 25-45 interested in premium skincare"
        },
        "other_examples": [
            {
                "original_text": "Download our new fitness app and transform your workout routine!",
                "target_tone": "fun",
                "target_platform": "instagram"
            },
            {
                "original_text": "Professional project management software for growing teams.",
                "target_tone": "authoritative",
                "target_platform": "linkedin"
            }
        ]
    }


@router.get(
    "/ad-rewriter/tone-guidelines",
    summary="Get Tone Guidelines",
    description="Get detailed guidelines for each available tone type"
)
async def get_tone_guidelines():
    """
    Get comprehensive tone guidelines.
    
    Returns detailed information about each available tone type including:
    - Description and characteristics
    - What to include and avoid
    - Best use cases and examples
    """
    return {
        "available_tones": {
            "professional": {
                "description": "Formal, credible, trustworthy, focused on expertise and results",
                "characteristics": [
                    "Formal language and proper grammar",
                    "Industry-specific terminology when appropriate",
                    "Credibility markers (certifications, awards, experience)",
                    "Data-driven statements and proof points",
                    "Authoritative and confident tone"
                ],
                "avoid": [
                    "Slang or colloquial expressions",
                    "Emojis or excessive punctuation",
                    "Casual or conversational language",
                    "Humor or playful elements"
                ],
                "best_for": [
                    "B2B marketing",
                    "Professional services",
                    "High-value products",
                    "LinkedIn campaigns",
                    "Industry publications"
                ],
                "example": "Leverage our proven expertise to optimize your marketing ROI with data-driven strategies that deliver measurable results."
            },
            "fun": {
                "description": "Playful, energetic, entertaining, engaging",
                "characteristics": [
                    "Casual and conversational language",
                    "Humor and wordplay",
                    "Emojis and visual elements",
                    "Exclamation points for emphasis",
                    "Creative and unexpected phrasing"
                ],
                "avoid": [
                    "Overly formal business language",
                    "Technical jargon",
                    "Serious or somber tone",
                    "Complex sentence structures"
                ],
                "best_for": [
                    "Consumer brands",
                    "Entertainment products",
                    "Social media campaigns",
                    "Younger demographics",
                    "Lifestyle brands"
                ],
                "example": "Ready to turn your marketing game from 'meh' to 'amazing'? 🚀 Let's make some magic happen!"
            },
            "casual": {
                "description": "Relaxed, conversational, approachable, friendly",
                "characteristics": [
                    "Conversational language",
                    "Contractions (you're, we'll, don't)",
                    "Friendly and approachable tone",
                    "Relatable everyday language",
                    "Personal pronouns (you, we, us)"
                ],
                "avoid": [
                    "Formal business jargon",
                    "Complex technical terms",
                    "Overly serious tone",
                    "Impersonal language"
                ],
                "best_for": [
                    "Consumer products",
                    "Community-focused brands",
                    "Social media content",
                    "Email marketing",
                    "Blog content"
                ],
                "example": "We get it - marketing can be overwhelming. That's why we're here to help you figure it out, step by step."
            },
            "urgent": {
                "description": "Time-sensitive, action-oriented, compelling, immediate",
                "characteristics": [
                    "Action-oriented language",
                    "Time pressure indicators",
                    "Strong call-to-action phrases",
                    "Urgency markers (now, today, limited time)",
                    "Direct and imperative sentences"
                ],
                "avoid": [
                    "Passive voice",
                    "Uncertain or hesitant language",
                    "Delayed action suggestions",
                    "Overly complex explanations"
                ],
                "best_for": [
                    "Sales promotions",
                    "Limited-time offers",
                    "Event marketing",
                    "Product launches",
                    "Conversion campaigns"
                ],
                "example": "Don't miss out! Limited spots available - secure your marketing transformation today before it's too late."
            },
            "friendly": {
                "description": "Warm, welcoming, supportive, personal",
                "characteristics": [
                    "Warm and welcoming language",
                    "Personal pronouns and direct address",
                    "Helpful and supportive tone",
                    "Inclusive language",
                    "Empathetic expressions"
                ],
                "avoid": [
                    "Cold or impersonal language",
                    "Aggressive sales tactics",
                    "Overly formal tone",
                    "Exclusionary language"
                ],
                "best_for": [
                    "Customer service",
                    "Community building",
                    "Support services",
                    "Educational content",
                    "Relationship marketing"
                ],
                "example": "We're here to support you every step of the way. Let's work together to achieve your marketing goals."
            },
            "authoritative": {
                "description": "Expert, confident, knowledgeable, trustworthy",
                "characteristics": [
                    "Confident and assertive statements",
                    "Expertise indicators and credentials",
                    "Proven results and testimonials",
                    "Leadership language",
                    "Definitive statements"
                ],
                "avoid": [
                    "Uncertain or hesitant language",
                    "Weak qualifiers (maybe, might, possibly)",
                    "Casual or informal tone",
                    "Self-deprecating statements"
                ],
                "best_for": [
                    "Thought leadership",
                    "Expert positioning",
                    "High-stakes decisions",
                    "Premium products",
                    "Industry authority"
                ],
                "example": "As the leading authority in marketing optimization, we guarantee results that transform your business performance."
            }
        },
        "selection_tips": [
            "Consider your target audience demographics and preferences",
            "Match tone to your brand personality and values",
            "Align with platform expectations and norms",
            "Test different tones with A/B testing",
            "Maintain consistency across your marketing channels"
        ]
    }


@router.get(
    "/ad-rewriter/platform-guidelines",
    summary="Get Platform Guidelines",
    description="Get optimization guidelines for each supported platform"
)
async def get_platform_guidelines():
    """
    Get comprehensive platform optimization guidelines.
    
    Returns detailed information about each supported platform including:
    - Character limits and technical constraints
    - Best practices and optimization tips
    - Audience characteristics and expectations
    - Call-to-action recommendations
    """
    return {
        "platforms": {
            "facebook": {
                "character_limit": 2200,
                "optimal_length": "125-150 characters for best engagement",
                "best_practices": [
                    "Focus on engaging storytelling",
                    "Build community and encourage interaction",
                    "Reference visual elements in the post",
                    "Include social proof and testimonials",
                    "Use clear, action-oriented language"
                ],
                "audience": "Diverse, social-focused users who value community and connection",
                "cta_recommendations": [
                    "Learn More",
                    "Shop Now",
                    "Sign Up",
                    "Get Started",
                    "Contact Us"
                ],
                "optimization_tips": [
                    "Include questions to encourage comments",
                    "Use Facebook-specific features (polls, events)",
                    "Optimize for mobile viewing",
                    "Test video vs. image content"
                ]
            },
            "instagram": {
                "character_limit": 2200,
                "optimal_length": "138-150 characters for captions",
                "best_practices": [
                    "Visual-first content approach",
                    "Strategic hashtag integration",
                    "Instagram Stories and Reels focus",
                    "Lifestyle and aspirational content",
                    "User-generated content incorporation"
                ],
                "audience": "Younger, visual-oriented users focused on lifestyle and aesthetics",
                "cta_recommendations": [
                    "Swipe up",
                    "Link in bio",
                    "DM us",
                    "Tag a friend",
                    "Share your story"
                ],
                "optimization_tips": [
                    "Use relevant hashtags (5-10 optimal)",
                    "Create Instagram-specific content formats",
                    "Leverage Instagram Shopping features",
                    "Focus on high-quality visuals"
                ]
            },
            "google": {
                "character_limit": 90,
                "optimal_length": "80-90 characters for headlines",
                "best_practices": [
                    "Keyword optimization for search relevance",
                    "Clear and specific value proposition",
                    "Highlight unique benefits and features",
                    "Include location for local relevance",
                    "Use numbers and specific details"
                ],
                "audience": "Intent-driven searchers looking for specific solutions",
                "cta_recommendations": [
                    "Get Quote",
                    "Call Now",
                    "Learn More",
                    "Download",
                    "Buy Online"
                ],
                "optimization_tips": [
                    "Include target keywords naturally",
                    "Match search intent closely",
                    "Use ad extensions effectively",
                    "Test different headline variations"
                ]
            },
            "linkedin": {
                "character_limit": 3000,
                "optimal_length": "150-300 characters for best engagement",
                "best_practices": [
                    "Professional and business-focused tone",
                    "Industry insights and thought leadership",
                    "Business value and ROI emphasis",
                    "Professional networking opportunities",
                    "B2B relationship building"
                ],
                "audience": "Business professionals, decision-makers, and industry experts",
                "cta_recommendations": [
                    "Connect with us",
                    "Request demo",
                    "Download whitepaper",
                    "Schedule consultation",
                    "Join our network"
                ],
                "optimization_tips": [
                    "Use professional language and terminology",
                    "Include relevant industry keywords",
                    "Leverage LinkedIn's professional features",
                    "Focus on business outcomes and results"
                ]
            },
            "twitter": {
                "character_limit": 280,
                "optimal_length": "71-100 characters for maximum engagement",
                "best_practices": [
                    "Concise and impactful messaging",
                    "Trending topics and hashtag usage",
                    "Real-time relevance and timeliness",
                    "Conversation starters and engagement",
                    "News and updates sharing"
                ],
                "audience": "News-focused, diverse users interested in real-time information",
                "cta_recommendations": [
                    "Retweet",
                    "Reply",
                    "Click link",
                    "Follow us",
                    "Join conversation"
                ],
                "optimization_tips": [
                    "Use relevant trending hashtags",
                    "Engage with current events and news",
                    "Keep messages concise and punchy",
                    "Include visual elements when possible"
                ]
            },
            "tiktok": {
                "character_limit": 150,
                "optimal_length": "100-150 characters",
                "best_practices": [
                    "Trend awareness and participation",
                    "Entertainment value and creativity",
                    "Authenticity and relatability",
                    "Viral potential and shareability",
                    "Quick, engaging content"
                ],
                "audience": "Gen Z and millennial users focused on entertainment and trends",
                "cta_recommendations": [
                    "Follow for more",
                    "Try this",
                    "Duet this",
                    "Share with friends",
                    "Comment below"
                ],
                "optimization_tips": [
                    "Stay current with TikTok trends",
                    "Use platform-specific language and slang",
                    "Create authentic, unpolished content",
                    "Encourage user interaction and participation"
                ]
            }
        },
        "cross_platform_tips": [
            "Adapt content for each platform's unique characteristics",
            "Maintain brand voice while respecting platform norms",
            "Test performance across different platforms",
            "Use platform-specific features and tools",
            "Monitor platform algorithm changes and adjust accordingly"
        ]
    }


@router.post(
    "/ad-rewriter/batch-rewrite",
    summary="Batch Rewrite Ad Text",
    description="Rewrite multiple ad texts with different tone and platform combinations",
    responses={
        200: {"description": "Batch rewriting completed successfully"},
        400: {"description": "Invalid request data"},
        500: {"description": "Internal server error"}
    }
)
async def batch_rewrite_ad_text(requests: List[AdRewriteRequest]) -> List[AdRewriteResponse]:
    """
    Rewrite multiple ad texts in a single request.
    
    This endpoint allows you to rewrite multiple pieces of ad text with different
    tone and platform combinations, useful for creating variations for A/B testing
    or multi-platform campaigns.
    
    Maximum 10 requests per batch to prevent timeouts.
    """
    try:
        if not requests:
            raise HTTPException(
                status_code=400,
                detail="No rewrite requests provided"
            )
        
        if len(requests) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 requests allowed per batch"
            )
        
        results = []
        for i, request in enumerate(requests):
            try:
                # Validate each request
                if not request.original_text or len(request.original_text.strip()) < 10:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Request {i+1}: Original text must be at least 10 characters long"
                    )
                
                # Perform rewriting
                result = await ad_rewriter_agent.rewrite_ad_text(request)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process batch request {i+1}: {e}")
                # Continue with other requests, but note the failure
                error_response = AdRewriteResponse(
                    original_text=request.original_text,
                    rewritten_text=f"Error: {str(e)}",
                    tone_applied=request.target_tone,
                    platform_optimized=request.target_platform,
                    improvements=["Failed to process this request"],
                    platform_specific_tips=["Please try again with valid input"],
                    alternative_versions=[]
                )
                results.append(error_response)
        
        logger.info(f"Batch ad text rewriting completed: {len(results)} requests processed")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process batch rewrite: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing batch rewrite"
        )


@router.get(
    "/ad-rewriter/sample-texts",
    summary="Get Sample Ad Texts",
    description="Get sample ad texts for testing the rewriter functionality"
)
async def get_sample_ad_texts():
    """
    Get sample ad texts for testing purposes.
    
    Returns various types of ad copy that can be used to test different
    tone and platform combinations.
    """
    return {
        "sample_texts": [
            {
                "type": "E-commerce Product",
                "original": "Our premium skincare line delivers results. Made with natural ingredients. Free shipping on orders over $50.",
                "suggested_tones": ["professional", "friendly", "casual"],
                "suggested_platforms": ["facebook", "instagram", "google"]
            },
            {
                "type": "SaaS Product",
                "original": "Streamline your workflow with our project management software. Increase productivity by 40%. Try it free for 30 days.",
                "suggested_tones": ["professional", "authoritative", "friendly"],
                "suggested_platforms": ["linkedin", "google", "facebook"]
            },
            {
                "type": "Event Promotion",
                "original": "Join us for the biggest marketing conference of the year. Learn from industry experts. Early bird pricing ends soon.",
                "suggested_tones": ["urgent", "professional", "fun"],
                "suggested_platforms": ["linkedin", "twitter", "facebook"]
            },
            {
                "type": "Service Business",
                "original": "Professional home cleaning services. Licensed and insured. Book your appointment today and get 20% off first cleaning.",
                "suggested_tones": ["professional", "friendly", "urgent"],
                "suggested_platforms": ["google", "facebook", "instagram"]
            },
            {
                "type": "Mobile App",
                "original": "Download our new fitness app. Track workouts, monitor progress, connect with friends. Available on iOS and Android.",
                "suggested_tones": ["fun", "casual", "friendly"],
                "suggested_platforms": ["instagram", "tiktok", "facebook"]
            }
        ],
        "testing_tips": [
            "Try the same text with different tones to see style variations",
            "Test platform-specific optimizations for the same content",
            "Use batch rewriting to create multiple versions quickly",
            "Compare alternative versions for A/B testing ideas"
        ]
    } 