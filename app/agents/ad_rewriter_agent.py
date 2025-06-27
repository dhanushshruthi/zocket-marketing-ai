"""
Ad Text Rewriter Agent

This agent rewrites user-uploaded ad text using different tones (professional, fun, etc.)
and optimizes it for different platforms using Azure OpenAI.
"""

import logging
from typing import List, Dict, Any, Optional

from app.models.schemas import (
    AdRewriteRequest, AdRewriteResponse, ToneType, PlatformType
)
from app.utils.azure_openai_client import get_azure_openai_client

logger = logging.getLogger(__name__)


class AdRewriterAgent:
    """Agent for rewriting ad text with different tones and platform optimization."""
    
    def __init__(self):
        self.azure_client = get_azure_openai_client()
        self.system_prompt = """
        You are an expert copywriter and digital marketing strategist specializing in creating compelling ad copy across various platforms and tones.
        
        Your expertise includes:
        1. Adapting tone and voice for different audiences
        2. Platform-specific optimization (Facebook, Instagram, Google, LinkedIn, etc.)
        3. Psychology of persuasion and conversion optimization
        4. A/B testing principles for ad copy
        5. Call-to-action optimization
        
        Always create engaging, conversion-focused copy that maintains brand consistency while adapting to platform requirements.
        Consider character limits, audience expectations, and platform best practices.
        """
        
        self.tone_guidelines = {
            ToneType.PROFESSIONAL: {
                "description": "Formal, credible, trustworthy, focused on expertise and results",
                "characteristics": ["formal language", "industry terminology", "credibility markers", "data-driven", "authoritative"],
                "avoid": ["slang", "emojis", "casual expressions", "humor"]
            },
            ToneType.FUN: {
                "description": "Playful, energetic, entertaining, engaging",
                "characteristics": ["casual language", "humor", "wordplay", "emojis", "exclamation points"],
                "avoid": ["overly formal language", "technical jargon", "serious tone"]
            },
            ToneType.CASUAL: {
                "description": "Relaxed, conversational, approachable, friendly",
                "characteristics": ["conversational language", "contractions", "friendly tone", "relatable"],
                "avoid": ["formal business language", "complex terminology"]
            },
            ToneType.URGENT: {
                "description": "Time-sensitive, action-oriented, compelling, immediate",
                "characteristics": ["action words", "time pressure", "strong CTAs", "urgency markers"],
                "avoid": ["passive language", "uncertain terms", "delayed action"]
            },
            ToneType.FRIENDLY: {
                "description": "Warm, welcoming, supportive, personal",
                "characteristics": ["warm language", "personal pronouns", "helpful tone", "inclusive"],
                "avoid": ["cold language", "impersonal tone", "aggressive sales language"]
            },
            ToneType.AUTHORITATIVE: {
                "description": "Expert, confident, knowledgeable, trustworthy",
                "characteristics": ["confident statements", "expertise indicators", "proven results", "leadership"],
                "avoid": ["uncertain language", "weak qualifiers", "casual tone"]
            }
        }
        
        self.platform_guidelines = {
            PlatformType.FACEBOOK: {
                "character_limit": 2200,
                "best_practices": ["engaging storytelling", "community focus", "visual elements mention", "social proof"],
                "cta_style": "action-oriented buttons",
                "audience": "diverse, social-focused"
            },
            PlatformType.INSTAGRAM: {
                "character_limit": 2200,
                "best_practices": ["visual-first content", "hashtag integration", "story elements", "lifestyle focus"],
                "cta_style": "swipe up, link in bio",
                "audience": "younger, visual-oriented"
            },
            PlatformType.GOOGLE: {
                "character_limit": 90,
                "best_practices": ["keyword optimization", "clear value proposition", "specific benefits", "local relevance"],
                "cta_style": "direct action words",
                "audience": "intent-driven searchers"
            },
            PlatformType.LINKEDIN: {
                "character_limit": 3000,
                "best_practices": ["professional tone", "industry insights", "business value", "networking focus"],
                "cta_style": "professional actions",
                "audience": "business professionals"
            },
            PlatformType.TWITTER: {
                "character_limit": 280,
                "best_practices": ["concise messaging", "trending topics", "real-time relevance", "conversation starters"],
                "cta_style": "retweet, reply, click",
                "audience": "news-focused, diverse"
            },
            PlatformType.TIKTOK: {
                "character_limit": 150,
                "best_practices": ["trend awareness", "entertainment value", "authenticity", "viral potential"],
                "cta_style": "engage, follow, try",
                "audience": "Gen Z, entertainment-focused"
            }
        }
    
    async def rewrite_ad_text(self, request: AdRewriteRequest) -> AdRewriteResponse:
        """Rewrite ad text with specified tone and platform optimization."""
        try:
            logger.info(f"Rewriting ad text for {request.target_tone} tone and {request.target_platform} platform")
            
            # Get guidelines for tone and platform
            tone_guide = self.tone_guidelines.get(request.target_tone, {})
            platform_guide = self.platform_guidelines.get(request.target_platform, {})
            
            # Generate the main rewritten text
            rewritten_text = await self._generate_rewritten_text(request, tone_guide, platform_guide)
            
            # Generate improvements list
            improvements = await self._analyze_improvements(request.original_text, rewritten_text, request)
            
            # Generate platform-specific tips
            platform_tips = await self._generate_platform_tips(request.target_platform, rewritten_text)
            
            # Generate alternative versions
            alternative_versions = await self._generate_alternatives(request, tone_guide, platform_guide)
            
            return AdRewriteResponse(
                original_text=request.original_text,
                rewritten_text=rewritten_text,
                tone_applied=request.target_tone,
                platform_optimized=request.target_platform,
                improvements=improvements,
                platform_specific_tips=platform_tips,
                alternative_versions=alternative_versions
            )
            
        except Exception as e:
            logger.error(f"Failed to rewrite ad text: {e}")
            raise
    
    async def _generate_rewritten_text(
        self, 
        request: AdRewriteRequest, 
        tone_guide: Dict[str, Any], 
        platform_guide: Dict[str, Any]
    ) -> str:
        """Generate the main rewritten ad text."""
        try:
            # Build detailed prompt
            prompt = f"""
            Rewrite the following ad text to match the specified tone and optimize for the target platform:
            
            Original Text: "{request.original_text}"
            
            Target Tone: {request.target_tone.value}
            - Description: {tone_guide.get('description', 'N/A')}
            - Characteristics: {', '.join(tone_guide.get('characteristics', []))}
            - Avoid: {', '.join(tone_guide.get('avoid', []))}
            
            Target Platform: {request.target_platform.value}
            - Character Limit: {platform_guide.get('character_limit', 'No specific limit')}
            - Best Practices: {', '.join(platform_guide.get('best_practices', []))}
            - Audience: {platform_guide.get('audience', 'General audience')}
            
            Additional Requirements:
            - Maximum Length: {request.max_length if request.max_length else 'No specific limit'}
            - Include CTA: {'Yes' if request.include_cta else 'No'}
            - Target Audience: {request.target_audience if request.target_audience else 'General audience'}
            
            Requirements:
            1. Maintain the core message and value proposition
            2. Apply the specified tone consistently
            3. Optimize for the target platform's best practices
            4. Ensure the text is engaging and conversion-focused
            5. Stay within character limits if specified
            
            Return only the rewritten ad text.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.7
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate rewritten text: {e}")
            raise
    
    async def _analyze_improvements(
        self, 
        original_text: str, 
        rewritten_text: str, 
        request: AdRewriteRequest
    ) -> List[str]:
        """Analyze and list the improvements made in the rewritten text."""
        try:
            prompt = f"""
            Compare the original ad text with the rewritten version and identify 4-6 specific improvements made:
            
            Original: "{original_text}"
            Rewritten: "{rewritten_text}"
            
            Target Tone: {request.target_tone.value}
            Target Platform: {request.target_platform.value}
            
            Identify improvements in:
            - Tone alignment
            - Platform optimization
            - Engagement factors
            - Conversion elements
            - Clarity and impact
            - Call-to-action effectiveness
            
            List specific improvements, one per line, starting with action words.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.3
            )
            
            # Parse improvements
            improvements = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit() or line[0].isupper()):
                    improvements.append(line.lstrip('-•0123456789. '))
            
            return improvements[:6]  # Limit to 6 improvements
            
        except Exception as e:
            logger.error(f"Failed to analyze improvements: {e}")
            return ["Optimized for target tone and platform"]
    
    async def _generate_platform_tips(self, platform: PlatformType, rewritten_text: str) -> List[str]:
        """Generate platform-specific optimization tips."""
        try:
            platform_guide = self.platform_guidelines.get(platform, {})
            
            prompt = f"""
            Provide 3-4 specific platform optimization tips for the following ad text on {platform.value}:
            
            Ad Text: "{rewritten_text}"
            
            Platform Characteristics:
            - Character Limit: {platform_guide.get('character_limit', 'No specific limit')}
            - Best Practices: {', '.join(platform_guide.get('best_practices', []))}
            - Audience: {platform_guide.get('audience', 'General audience')}
            - CTA Style: {platform_guide.get('cta_style', 'Standard CTAs')}
            
            Focus on:
            - Platform-specific features to leverage
            - Audience behavior considerations
            - Technical optimizations
            - Performance enhancement suggestions
            
            Provide actionable tips, one per line.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.4
            )
            
            # Parse tips
            tips = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit() or len(line) > 10):
                    tips.append(line.lstrip('-•0123456789. '))
            
            return tips[:4]  # Limit to 4 tips
            
        except Exception as e:
            logger.error(f"Failed to generate platform tips: {e}")
            return [f"Optimized for {platform.value} best practices"]
    
    async def _generate_alternatives(
        self, 
        request: AdRewriteRequest, 
        tone_guide: Dict[str, Any], 
        platform_guide: Dict[str, Any]
    ) -> List[str]:
        """Generate alternative versions of the rewritten text."""
        try:
            prompt = f"""
            Create 2-3 alternative versions of rewritten ad text for A/B testing:
            
            Original Text: "{request.original_text}"
            Target Tone: {request.target_tone.value}
            Target Platform: {request.target_platform.value}
            
            Create variations that:
            1. Maintain the same tone and platform optimization
            2. Test different approaches (emotional vs rational, benefit vs feature focus, etc.)
            3. Vary the call-to-action style
            4. Experiment with different hooks or value propositions
            
            Return each alternative on a separate line, numbered.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.8  # Higher temperature for more creative variations
            )
            
            # Parse alternatives
            alternatives = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    alternatives.append(line[2:].strip())
                elif line and len(line) > 20 and not line.startswith(('Original', 'Target', 'Create')):
                    alternatives.append(line)
            
            return alternatives[:3]  # Limit to 3 alternatives
            
        except Exception as e:
            logger.error(f"Failed to generate alternatives: {e}")
            return [] 