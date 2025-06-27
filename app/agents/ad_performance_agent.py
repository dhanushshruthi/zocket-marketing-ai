"""
Ad Performance Analyzer Agent

This agent reviews Meta/Google ad performance CSVs and outputs insights 
and creative improvement suggestions using Azure OpenAI.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from io import StringIO

from app.models.schemas import AdPerformanceData, AdPerformanceResponse
from app.utils.azure_openai_client import get_azure_openai_client

logger = logging.getLogger(__name__)


class AdPerformanceAgent:
    """Agent for analyzing ad performance data and providing insights."""
    
    def __init__(self):
        self.azure_client = get_azure_openai_client()
        self.system_prompt = """
        You are an expert digital marketing analyst specializing in Meta (Facebook/Instagram) and Google Ads performance analysis.
        
        Your role is to:
        1. Analyze ad campaign performance data
        2. Identify key insights and trends
        3. Provide actionable recommendations for improvement
        4. Suggest creative optimizations
        5. Highlight top performers and underperformers
        
        Focus on practical, data-driven recommendations that can improve campaign ROI.
        Use clear, professional language and provide specific, actionable advice.
        """
    
    async def analyze_performance(self, ad_data: List[AdPerformanceData], analysis_type: str = "comprehensive") -> AdPerformanceResponse:
        """Analyze ad performance data and generate insights."""
        try:
            logger.info(f"Analyzing {len(ad_data)} ad campaigns")
            
            df = self._prepare_dataframe(ad_data)
            
            metrics = self._calculate_metrics(df)
            
            insights = await self._generate_insights(df, metrics, analysis_type)
            
            recommendations = await self._generate_recommendations(df, metrics)
            
            top_performers, underperformers = self._identify_performance_tiers(df)
            
            summary = self._create_summary(df, metrics)
            
            return AdPerformanceResponse(
                summary=summary,
                insights=insights,
                recommendations=recommendations,
                top_performers=top_performers,
                underperformers=underperformers,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze ad performance: {e}")
            raise
    
    def _prepare_dataframe(self, ad_data: List[AdPerformanceData]) -> pd.DataFrame:
        """Convert ad data to pandas DataFrame with calculated metrics."""
        data = []
        for ad in ad_data:
            row = {
                'campaign_name': ad.campaign_name,
                'impressions': ad.impressions,
                'clicks': ad.clicks,
                'conversions': ad.conversions,
                'spend': ad.spend,
                'ctr': ad.ctr if ad.ctr is not None else (ad.clicks / ad.impressions * 100 if ad.impressions > 0 else 0),
                'cpa': ad.cpa if ad.cpa is not None else (ad.spend / ad.conversions if ad.conversions > 0 else 0),
                'roas': ad.roas if ad.roas is not None else (0 if ad.spend == 0 else 0)  # Revenue data needed for ROAS
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        df['conversion_rate'] = (df['conversions'] / df['clicks'] * 100).fillna(0)
        df['cpm'] = (df['spend'] / df['impressions'] * 1000).fillna(0)
        df['cost_per_click'] = (df['spend'] / df['clicks']).fillna(0)
        
        return df
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate aggregate metrics."""
        total_impressions = df['impressions'].sum()
        total_clicks = df['clicks'].sum()
        total_conversions = df['conversions'].sum()
        total_spend = df['spend'].sum()
        
        return {
            'total_impressions': float(total_impressions),
            'total_clicks': float(total_clicks),
            'total_conversions': float(total_conversions),
            'total_spend': float(total_spend),
            'overall_ctr': float(total_clicks / total_impressions * 100 if total_impressions > 0 else 0),
            'overall_conversion_rate': float(total_conversions / total_clicks * 100 if total_clicks > 0 else 0),
            'overall_cpa': float(total_spend / total_conversions if total_conversions > 0 else 0),
            'overall_cpm': float(total_spend / total_impressions * 1000 if total_impressions > 0 else 0),
            'average_cpc': float(total_spend / total_clicks if total_clicks > 0 else 0),
            'num_campaigns': len(df)
        }
    
    async def _generate_insights(self, df: pd.DataFrame, metrics: Dict[str, float], analysis_type: str) -> List[str]:
        """Generate AI-powered insights from the data."""
        try:
            data_summary = f"""
            Performance Data Analysis:
            - Total Campaigns: {metrics['num_campaigns']}
            - Total Spend: ${metrics['total_spend']:,.2f}
            - Total Impressions: {metrics['total_impressions']:,.0f}
            - Total Clicks: {metrics['total_clicks']:,.0f}
            - Total Conversions: {metrics['total_conversions']:,.0f}
            - Overall CTR: {metrics['overall_ctr']:.2f}%
            - Overall Conversion Rate: {metrics['overall_conversion_rate']:.2f}%
            - Overall CPA: ${metrics['overall_cpa']:.2f}
            - Overall CPM: ${metrics['overall_cpm']:.2f}
            - Average CPC: ${metrics['average_cpc']:.2f}
            
            Campaign Performance Breakdown:
            {df.to_string(index=False)}
            """
            
            prompt = f"""
            Analyze the following ad performance data and provide 5-7 key insights:
            
            {data_summary}
            
            Focus on:
            - Performance patterns and trends
            - Cost efficiency analysis
            - Conversion optimization opportunities
            - Budget allocation insights
            - Creative performance indicators
            
            Provide specific, actionable insights based on the data.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.3
            )
            
            insights = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    insights.append(line.lstrip('-•0123456789. '))
            
            return insights[:7] 
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return ["Failed to generate AI insights due to an error."]
    
    async def _generate_recommendations(self, df: pd.DataFrame, metrics: Dict[str, float]) -> List[str]:
        """Generate AI-powered recommendations."""
        try:
            high_cpa_campaigns = df[df['cpa'] > df['cpa'].quantile(0.75)]['campaign_name'].tolist()
            low_ctr_campaigns = df[df['ctr'] < df['ctr'].quantile(0.25)]['campaign_name'].tolist()
            low_conversion_campaigns = df[df['conversion_rate'] < df['conversion_rate'].quantile(0.25)]['campaign_name'].tolist()
            
            problem_analysis = f"""
            Problem Areas Identified:
            - High CPA Campaigns: {', '.join(high_cpa_campaigns[:3])}
            - Low CTR Campaigns: {', '.join(low_ctr_campaigns[:3])}
            - Low Conversion Rate Campaigns: {', '.join(low_conversion_campaigns[:3])}
            
            Overall Metrics:
            - Average CPA: ${df['cpa'].mean():.2f}
            - Average CTR: {df['ctr'].mean():.2f}%
            - Average Conversion Rate: {df['conversion_rate'].mean():.2f}%
            """
            
            prompt = f"""
            Based on the following ad performance analysis, provide 5-7 specific, actionable recommendations to improve campaign performance:
            
            {problem_analysis}
            
            Focus on:
            - Cost optimization strategies
            - Creative improvement suggestions
            - Targeting refinements
            - Budget reallocation recommendations
            - A/B testing opportunities
            
            Provide concrete, implementable recommendations.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.4
            )

            recommendations = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    recommendations.append(line.lstrip('-•0123456789. '))
            
            return recommendations[:7] 
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Failed to generate AI recommendations due to an error."]
    
    def _identify_performance_tiers(self, df: pd.DataFrame) -> tuple[List[str], List[str]]:
        """Identify top performers and underperformers."""
        df['performance_score'] = (
            (df['ctr'] / df['ctr'].max() * 0.3) +
            (df['conversion_rate'] / df['conversion_rate'].max() * 0.4) +
            ((df['cpa'].max() - df['cpa']) / df['cpa'].max() * 0.3) 
        ).fillna(0)
        
        df_sorted = df.sort_values('performance_score', ascending=False)
        
        num_campaigns = len(df)
        top_count = max(1, num_campaigns // 5)
        bottom_count = max(1, num_campaigns // 5)
        
        top_performers = df_sorted.head(top_count)['campaign_name'].tolist()
        underperformers = df_sorted.tail(bottom_count)['campaign_name'].tolist()
        
        return top_performers, underperformers
    
    def _create_summary(self, df: pd.DataFrame, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create performance summary."""
        return {
            'total_campaigns': metrics['num_campaigns'],
            'total_spend': metrics['total_spend'],
            'total_impressions': metrics['total_impressions'],
            'total_clicks': metrics['total_clicks'],
            'total_conversions': metrics['total_conversions'],
            'overall_performance': {
                'ctr': round(metrics['overall_ctr'], 2),
                'conversion_rate': round(metrics['overall_conversion_rate'], 2),
                'cpa': round(metrics['overall_cpa'], 2),
                'cpm': round(metrics['overall_cpm'], 2),
                'cpc': round(metrics['average_cpc'], 2)
            },
            'campaign_performance_distribution': {
                'best_ctr': round(df['ctr'].max(), 2),
                'worst_ctr': round(df['ctr'].min(), 2),
                'best_conversion_rate': round(df['conversion_rate'].max(), 2),
                'worst_conversion_rate': round(df['conversion_rate'].min(), 2),
                'lowest_cpa': round(df['cpa'].min(), 2),
                'highest_cpa': round(df['cpa'].max(), 2)
            }
        } 