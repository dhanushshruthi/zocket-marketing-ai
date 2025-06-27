"""
Test Data Generation and Management for Marketing AI Agents Evaluation

This module provides test data generation, management, and validation
for evaluating marketing AI agents across different scenarios.
"""

import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

class TestDataGenerator:
    """Generate test data for evaluating marketing AI agents"""
    
    def __init__(self, seed: int = 42):
        """Initialize test data generator with seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Marketing domain data
        self.platforms = ["facebook", "instagram", "google", "linkedin", "twitter", "tiktok"]
        self.tones = ["professional", "casual", "fun", "urgent", "friendly", "authoritative"]
        self.industries = ["ecommerce", "saas", "fashion", "finance", "healthcare", "education"]
        self.marketing_topics = [
            "facebook advertising", "content marketing", "email campaigns", 
            "social media strategy", "influencer marketing", "video marketing",
            "conversion optimization", "brand awareness", "lead generation"
        ]
    
    def generate_ad_performance_test_cases(self, num_cases: int = 50) -> List[Dict[str, Any]]:
        """Generate test cases for Ad Performance Analyzer"""
        test_cases = []
        
        for i in range(num_cases):
            # Generate realistic ad performance data
            impressions = random.randint(1000, 100000)
            ctr = random.uniform(0.005, 0.08)  # 0.5% to 8% CTR
            clicks = int(impressions * ctr)
            conversion_rate = random.uniform(0.01, 0.15)  # 1% to 15% conversion rate
            conversions = int(clicks * conversion_rate)
            cpc = random.uniform(0.5, 5.0)
            spend = clicks * cpc
            
            ad_data = [
                {
                    "campaign_name": f"Campaign_{i}_{random.choice(self.industries)}",
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": round(spend, 2),
                    "platform": random.choice(self.platforms),
                    "industry": random.choice(self.industries)
                }
            ]
            
            # Generate expected insights
            expected_insights = {
                "ctr": round(ctr, 4),
                "conversion_rate": round(conversion_rate, 4),
                "cpc": round(cpc, 2),
                "cpa": round(spend / conversions if conversions > 0 else 0, 2),
                "roas": round(conversions * 50 / spend if spend > 0 else 0, 2)  # Assuming $50 average order value
            }
            
            test_cases.append({
                "test_case_id": i,
                "ad_data": ad_data,
                "expected_metrics": expected_insights,
                "difficulty": "medium" if ctr > 0.02 else "hard"
            })
        
        return test_cases
    
    def generate_blog_search_test_cases(self, num_cases: int = 30) -> List[Dict[str, Any]]:
        """Generate test cases for Blog Search Agent"""
        test_cases = []
        
        # Predefined queries with expected results
        query_templates = [
            ("How to improve {topic} for {industry}?", ["strategy", "optimization", "best practices"]),
            ("Best {topic} strategies for {platform}", ["platform", "strategies", "marketing"]),
            ("What are the latest trends in {topic}?", ["trends", "latest", "updates"]),
            ("{topic} case studies and examples", ["case studies", "examples", "success stories"]),
            ("Common mistakes in {topic} campaigns", ["mistakes", "errors", "avoid"])
        ]
        
        for i in range(num_cases):
            template, expected_keywords = random.choice(query_templates)
            topic = random.choice(self.marketing_topics)
            platform = random.choice(self.platforms)
            industry = random.choice(self.industries)
            
            query = template.format(topic=topic, platform=platform, industry=industry)
            
            # Generate ground truth answer
            ground_truth = f"Based on industry best practices, {topic} for {industry} companies should focus on..."
            
            # Generate expected documents
            expected_docs = [
                f"doc_{topic.replace(' ', '_')}_{i}",
                f"article_{platform}_{topic.replace(' ', '_')}",
                f"guide_{industry}_{topic.replace(' ', '_')}"
            ]
            
            test_cases.append({
                "test_case_id": i,
                "query": query,
                "expected_keywords": expected_keywords,
                "ground_truth_answer": ground_truth,
                "expected_documents": expected_docs,
                "topic": topic,
                "platform": platform,
                "industry": industry
            })
        
        return test_cases
    
    def generate_ad_rewriter_test_cases(self, num_cases: int = 40) -> List[Dict[str, Any]]:
        """Generate test cases for Ad Text Rewriter"""
        test_cases = []
        
        original_texts = [
            "Buy our amazing product now! Limited time offer.",
            "Transform your business with our innovative solution.",
            "Join thousands of satisfied customers today.",
            "Get 50% off your first purchase this week only.",
            "Experience the difference with our premium service.",
            "Don't miss out on this exclusive opportunity.",
            "Boost your productivity with our cutting-edge tool.",
            "Save time and money with our efficient platform."
        ]
        
        for i in range(num_cases):
            original_text = random.choice(original_texts)
            target_tone = random.choice(self.tones)
            target_platform = random.choice(self.platforms)
            
            # Generate expected rewrite based on tone and platform
            expected_rewrite = self._generate_expected_rewrite(original_text, target_tone, target_platform)
            
            test_cases.append({
                "test_case_id": i,
                "original_text": original_text,
                "target_tone": target_tone,
                "target_platform": target_platform,
                "expected_rewrite": expected_rewrite,
                "include_cta": random.choice([True, False])
            })
        
        return test_cases
    
    def generate_web_crawler_test_cases(self, num_cases: int = 20) -> List[Dict[str, Any]]:
        """Generate test cases for Web Crawler"""
        test_cases = []
        
        # Sample marketing blog URLs and expected content
        blog_urls = [
            "https://blog.hubspot.com/marketing/facebook-advertising-guide",
            "https://blog.hootsuite.com/social-media-marketing-strategy",
            "https://contentmarketinginstitute.com/blog/content-marketing-strategy",
            "https://blog.marketo.com/email-marketing-best-practices",
            "https://buffer.com/library/social-media-advertising-tips"
        ]
        
        for i in range(num_cases):
            url = random.choice(blog_urls)
            
            # Extract expected title and keywords from URL
            url_parts = url.split('/')[-1].split('-')
            expected_title = ' '.join(word.capitalize() for word in url_parts[:3])
            expected_keywords = url_parts[:5]
            
            test_cases.append({
                "test_case_id": i,
                "url": url,
                "expected_title": expected_title,
                "expected_keywords": expected_keywords,
                "expected_content_length": random.randint(500, 3000),
                "should_extract_date": True,
                "should_extract_author": True
            })
        
        return test_cases
    
    def generate_comprehensive_test_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate comprehensive test suite for all agents"""
        return {
            "ad_performance": self.generate_ad_performance_test_cases(3),
            "blog_search": self.generate_blog_search_test_cases(3),
            "ad_rewriter": self.generate_ad_rewriter_test_cases(3)
            # web_crawler removed as requested
        }
    
    def save_test_suite(self, test_suite: Dict[str, List[Dict[str, Any]]], filename: str = "test_suite.json"):
        """Save test suite to JSON file"""
        test_suite_with_metadata = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_test_cases": sum(len(cases) for cases in test_suite.values()),
                "agents": list(test_suite.keys())
            },
            "test_cases": test_suite
        }
        
        with open(filename, 'w') as f:
            json.dump(test_suite_with_metadata, f, indent=2)
    
    def load_test_suite(self, filename: str = "test_suite.json") -> Dict[str, List[Dict[str, Any]]]:
        """Load test suite from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return data.get("test_cases", {})
    
    def _generate_expected_rewrite(self, original_text: str, tone: str, platform: str) -> str:
        """Generate expected rewrite based on tone and platform"""
        # Tone modifications
        tone_modifiers = {
            "professional": "We invite you to explore our solution",
            "casual": "Check out our awesome product",
            "fun": "Ready to have some fun with our amazing product?",
            "urgent": "Act now! Don't miss this incredible opportunity",
            "friendly": "We'd love to help you with our great product",
            "authoritative": "Industry leaders choose our proven solution"
        }
        
        # Platform-specific modifications
        platform_modifiers = {
            "facebook": "Share with friends!",
            "instagram": "📱 Swipe up for more!",
            "google": "Learn more about our solution",
            "linkedin": "Connect with industry professionals",
            "twitter": "Join the conversation #marketing",
            "tiktok": "🎵 Trending now!"
        }
        
        base_rewrite = tone_modifiers.get(tone, original_text)
        platform_addition = platform_modifiers.get(platform, "")
        
        return f"{base_rewrite} {platform_addition}".strip()

class BenchmarkDatasets:
    """Predefined benchmark datasets for consistent evaluation"""
    
    @staticmethod
    def get_marketing_relevance_benchmark() -> List[Dict[str, Any]]:
        """Get benchmark dataset for relevance evaluation"""
        return [
            {
                "query": "Facebook advertising best practices",
                "relevant_response": "Facebook advertising best practices include targeting specific audiences, using compelling visuals, A/B testing ad copy, and optimizing for mobile users.",
                "irrelevant_response": "The weather today is sunny and warm, perfect for outdoor activities and spending time with family.",
                "expected_relevance_score": 0.85
            },
            {
                "query": "Email marketing conversion rates",
                "relevant_response": "Email marketing conversion rates typically range from 2-5% across industries, with personalized emails achieving higher conversion rates.",
                "irrelevant_response": "Cooking recipes require careful attention to ingredients and cooking times for best results.",
                "expected_relevance_score": 0.90
            }
        ]
    
    @staticmethod
    def get_hallucination_detection_benchmark() -> List[Dict[str, Any]]:
        """Get benchmark dataset for hallucination detection"""
        return [
            {
                "response": "Facebook ads have a 2.5% average click-through rate according to recent studies.",
                "ground_truth": ["Facebook ads typically have click-through rates between 0.5% and 3%"],
                "expected_hallucination_rate": 0.0
            },
            {
                "response": "Email marketing has a 99% open rate and all emails are guaranteed to reach the inbox.",
                "ground_truth": ["Email marketing open rates average 20-25% across industries"],
                "expected_hallucination_rate": 0.8
            }
        ]
    
    @staticmethod
    def get_extraction_benchmark() -> List[Dict[str, Any]]:
        """Get benchmark dataset for entity extraction"""
        return [
            {
                "predicted_entities": ["Facebook", "Instagram", "Google Ads", "LinkedIn"],
                "true_entities": ["Facebook", "Instagram", "Google Ads", "Twitter"],
                "expected_f1_score": 0.67
            },
            {
                "predicted_entities": ["email marketing", "social media", "content marketing"],
                "true_entities": ["email marketing", "social media", "content marketing"],
                "expected_f1_score": 1.0
            }
        ] 