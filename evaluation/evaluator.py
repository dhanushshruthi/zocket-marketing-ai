"""
Main Evaluator for Marketing AI Agents

This module provides the main evaluation interface for testing and benchmarking
all marketing AI agents across different metrics and scenarios.
"""

import asyncio
import json
import logging
import sys
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .metrics import EvaluationMetrics
from app.agents.ad_performance_agent import AdPerformanceAgent
from app.agents.blog_search_agent import BlogSearchAgent
from app.agents.ad_rewriter_agent import AdRewriterAgent
from app.utils.web_crawl import WebCrawler
from app.database.chroma_client import ChromaClient
from app.models.schemas import (
    AdPerformanceData, BlogSearchRequest, AdRewriteRequest,
    ToneType, PlatformType
)

logger = logging.getLogger(__name__)

class MarketingAgentEvaluator:
    """Main evaluator for all marketing AI agents"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize the evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.metrics = EvaluationMetrics()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize actual agents
        self.ad_performance_agent = AdPerformanceAgent()
        self.chroma_client = ChromaClient()
        self.blog_search_agent = BlogSearchAgent(self.chroma_client)
        self.ad_rewriter_agent = AdRewriterAgent()
        self.web_crawler = WebCrawler(delay=1.0)
        
        # Initialize results storage
        self.results = {
            "ad_performance": [],
            "blog_search": [],
            "ad_rewriter": [],
            "web_crawler": []
        }
    
    async def evaluate_ad_performance_agent(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the Ad Performance Analyzer agent
        
        Args:
            test_cases: List of test cases with ad data and expected insights
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating Ad Performance Analyzer agent...")
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                # Extract test data
                ad_data_raw = test_case.get("ad_data", [])
                expected_insights = test_case.get("expected_insights", {})
                expected_metrics = test_case.get("expected_metrics", {})
                
                # Convert raw ad data to AdPerformanceData objects
                ad_data = []
                for ad in ad_data_raw:
                    ad_obj = AdPerformanceData(
                        campaign_name=ad.get("campaign_name", f"Campaign_{i}"),
                        impressions=ad.get("impressions", 0),
                        clicks=ad.get("clicks", 0),
                        conversions=ad.get("conversions", 0),
                        spend=ad.get("spend", 0.0)
                    )
                    ad_data.append(ad_obj)
                
                # Call actual Ad Performance Agent
                logger.info(f"Calling Ad Performance Agent for test case {i}")
                agent_response = await self.ad_performance_agent.analyze_performance(ad_data)
                
                result = {
                    "test_case_id": i,
                    "timestamp": datetime.now().isoformat(),
                    "input_size": len(ad_data),
                    "agent_response": {
                        "summary": agent_response.summary,
                        "insights": agent_response.insights,
                        "recommendations": agent_response.recommendations,
                        "top_performers": agent_response.top_performers,
                        "metrics": agent_response.metrics
                    }
                }
                
                # Evaluate metrics accuracy if provided
                if expected_metrics:
                    predicted_metrics = agent_response.metrics
                    metrics_evaluation = self.metrics.evaluate_ad_performance_insights(
                        predicted_insights=predicted_metrics,
                        ground_truth_insights=expected_metrics
                    )
                    result.update(metrics_evaluation)
                
                # Evaluate insight quality using ROUGE scores
                if agent_response.insights:
                    insights_text = " ".join(agent_response.insights)
                    if "reference_insights" in test_case:
                        rouge_scores = self.metrics.calculate_rouge_scores(
                            reference=test_case["reference_insights"],
                            summary=insights_text
                        )
                        result["rouge_scores"] = rouge_scores
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case {i}: {e}")
                results.append({"test_case_id": i, "error": str(e)})
        
        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(results, "ad_performance")
        
        # Store results
        self.results["ad_performance"] = results
        
        evaluation_result = {
            "individual_results": results,
            "aggregate_metrics": aggregate_results,
            "total_test_cases": len(test_cases),
            "successful_evaluations": len([r for r in results if "error" not in r])
        }
        
        # Save results immediately after this agent
        await self._save_agent_results("ad_performance", evaluation_result)
        
        return evaluation_result
    
    async def evaluate_blog_search_agent(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the Blog Search agent
        
        Args:
            test_cases: List of test cases with queries and expected results
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating Blog Search agent...")
        
        results = []
        
        # Initialize ChromaDB for blog search
        await self.chroma_client.initialize()
        
        for i, test_case in enumerate(test_cases):
            try:
                query = test_case.get("query", "")
                expected_docs = test_case.get("expected_documents", [])
                ground_truth = test_case.get("ground_truth_answer", "")
                
                # Create blog search request
                search_request = BlogSearchRequest(
                    query=query,
                    max_results=5,
                    filter_category=test_case.get("filter_category"),
                    filter_topic=test_case.get("filter_topic")
                )
                
                # Call actual Blog Search Agent
                logger.info(f"Calling Blog Search Agent for test case {i}: {query}")
                try:
                    agent_response = await self.blog_search_agent.search_and_answer(search_request)
                    
                    # Extract response text from results
                    response_text = ""
                    retrieved_doc_ids = []
                    
                    if agent_response.results:
                        response_text = " ".join([result.content[:200] for result in agent_response.results[:3]])
                        retrieved_doc_ids = [result.id for result in agent_response.results]
                    
                    result = {
                        "test_case_id": i,
                        "timestamp": datetime.now().isoformat(),
                        "query": query,
                        "agent_response": {
                            "total_results": agent_response.total_results,
                            "search_time_ms": agent_response.search_time_ms,
                            "suggested_queries": agent_response.suggested_queries,
                            "results_summary": response_text[:500]  # Limit for storage
                        }
                    }
                    
                    # Calculate relevance scores
                    if response_text:
                        context_docs = [result.content for result in agent_response.results[:3]]
                        relevance_scores = self.metrics.calculate_relevance_score(
                            query=query,
                            response=response_text,
                            context=context_docs
                        )
                        result["relevance_scores"] = relevance_scores
                    
                    # Calculate hallucination detection
                    if ground_truth and response_text:
                        hallucination_results = self.metrics.detect_hallucination(
                            response=response_text,
                            ground_truth=[ground_truth]
                        )
                        result["hallucination_detection"] = hallucination_results
                    
                    # Evaluate document retrieval
                    if expected_docs and retrieved_doc_ids:
                        retrieval_f1 = self.metrics.calculate_extraction_f1(
                            predicted_entities=retrieved_doc_ids,
                            true_entities=expected_docs
                        )
                        result["document_retrieval_f1"] = retrieval_f1
                    
                except Exception as agent_error:
                    logger.warning(f"Agent call failed for test case {i}: {agent_error}")
                    # Fallback to basic evaluation
                    result = {
                        "test_case_id": i,
                        "timestamp": datetime.now().isoformat(),
                        "query": query,
                        "agent_error": str(agent_error)
                    }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case {i}: {e}")
                results.append({"test_case_id": i, "error": str(e)})
        
        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(results, "blog_search")
        
        # Store results
        self.results["blog_search"] = results
        
        evaluation_result = {
            "individual_results": results,
            "aggregate_metrics": aggregate_results,
            "total_test_cases": len(test_cases),
            "successful_evaluations": len([r for r in results if "error" not in r])
        }
        
        # Save results immediately after this agent
        await self._save_agent_results("blog_search", evaluation_result)
        
        return evaluation_result
    
    async def evaluate_ad_rewriter_agent(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the Ad Text Rewriter agent
        
        Args:
            test_cases: List of test cases with original text and rewrite parameters
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating Ad Text Rewriter agent...")
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                original_text = test_case.get("original_text", "")
                target_tone = test_case.get("target_tone", "professional")
                target_platform = test_case.get("target_platform", "facebook")
                expected_rewrite = test_case.get("expected_rewrite", "")
                
                # Convert tone and platform strings to enums
                try:
                    tone_enum = ToneType(target_tone.lower())
                except ValueError:
                    tone_enum = ToneType.PROFESSIONAL
                    
                try:
                    platform_enum = PlatformType(target_platform.lower())
                except ValueError:
                    platform_enum = PlatformType.FACEBOOK
                
                # Create ad rewrite request
                rewrite_request = AdRewriteRequest(
                    original_text=original_text,
                    target_tone=tone_enum,
                    target_platform=platform_enum,
                    include_cta=test_case.get("include_cta", True),
                    target_audience=test_case.get("target_audience"),
                    max_length=test_case.get("max_length")
                )
                
                # Call actual Ad Rewriter Agent
                logger.info(f"Calling Ad Rewriter Agent for test case {i}: {target_tone} tone, {target_platform} platform")
                agent_response = await self.ad_rewriter_agent.rewrite_ad_text(rewrite_request)
                
                result = {
                    "test_case_id": i,
                    "timestamp": datetime.now().isoformat(),
                    "original_text": original_text,
                    "target_tone": target_tone,
                    "target_platform": target_platform,
                    "agent_response": {
                        "rewritten_text": agent_response.rewritten_text,
                        "improvements": agent_response.improvements,
                        "platform_tips": agent_response.platform_specific_tips,
                        "alternative_versions": agent_response.alternative_versions
                    }
                }
                
                # Evaluate content quality
                content_quality = self.metrics.evaluate_content_quality(agent_response.rewritten_text)
                result["content_quality"] = content_quality
                
                # Calculate BLEU score if expected rewrite is provided
                if expected_rewrite:
                    bleu_score = self.metrics.calculate_bleu_score(
                        reference=expected_rewrite,
                        candidate=agent_response.rewritten_text
                    )
                    result["bleu_score"] = bleu_score
                
                # Calculate semantic similarity with original
                relevance_to_original = self.metrics.calculate_relevance_score(
                    query=original_text,
                    response=agent_response.rewritten_text
                )
                result["semantic_preservation"] = relevance_to_original
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case {i}: {e}")
                results.append({"test_case_id": i, "error": str(e)})
        
        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(results, "ad_rewriter")
        
        # Store results
        self.results["ad_rewriter"] = results
        
        evaluation_result = {
            "individual_results": results,
            "aggregate_metrics": aggregate_results,
            "total_test_cases": len(test_cases),
            "successful_evaluations": len([r for r in results if "error" not in r])
        }
        
        # Save results immediately after this agent
        await self._save_agent_results("ad_rewriter", evaluation_result)
        
        return evaluation_result
    
    async def evaluate_web_crawler(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the Web Crawler functionality
        
        Args:
            test_cases: List of test cases with URLs and expected extractions
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating Web Crawler...")
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                url = test_case.get("url", "")
                expected_title = test_case.get("expected_title", "")
                expected_content_keywords = test_case.get("expected_keywords", [])
                
                # Call actual Web Crawler
                logger.info(f"Calling Web Crawler for test case {i}: {url}")
                try:
                    # Scrape the page
                    document = self.web_crawler.scrape_page(url)
                    
                    if document:
                        extracted_title = document['metadata'].get('title', '')
                        extracted_content = document['content']
                        
                        # Simple keyword extraction from content (first 10 most common words)
                        words = re.findall(r'\b\w+\b', extracted_content.lower())
                        word_freq = {}
                        for word in words:
                            if len(word) > 3:  # Skip short words
                                word_freq[word] = word_freq.get(word, 0) + 1
                        extracted_keywords = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
                        
                        result = {
                            "test_case_id": i,
                            "timestamp": datetime.now().isoformat(),
                            "url": url,
                            "extracted_title": extracted_title,
                            "content_length": len(extracted_content),
                            "extracted_keywords_count": len(extracted_keywords),
                            "agent_response": {
                                "document_id": document['id'],
                                "content_preview": extracted_content[:200] + "...",
                                "metadata": document['metadata']
                            }
                        }
                        
                        # Evaluate title extraction
                        if expected_title and extracted_title:
                            title_similarity = self.metrics.calculate_relevance_score(
                                query=expected_title,
                                response=extracted_title
                            )
                            result["title_extraction_accuracy"] = title_similarity
                        
                        # Evaluate keyword extraction
                        if expected_content_keywords:
                            keyword_f1 = self.metrics.calculate_extraction_f1(
                                predicted_entities=extracted_keywords,
                                true_entities=expected_content_keywords
                            )
                            result["keyword_extraction_f1"] = keyword_f1
                        
                        # Evaluate content quality
                        content_quality = self.metrics.evaluate_content_quality(extracted_content)
                        result["content_quality"] = content_quality
                        
                    else:
                        # Crawling failed
                        result = {
                            "test_case_id": i,
                            "timestamp": datetime.now().isoformat(),
                            "url": url,
                            "crawl_failed": True,
                            "error": "Failed to extract content from URL"
                        }
                
                except Exception as crawler_error:
                    logger.warning(f"Web crawler failed for test case {i}: {crawler_error}")
                    result = {
                        "test_case_id": i,
                        "timestamp": datetime.now().isoformat(),
                        "url": url,
                        "crawler_error": str(crawler_error)
                    }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating test case {i}: {e}")
                results.append({"test_case_id": i, "error": str(e)})
        
        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(results, "web_crawler")
        
        # Store results
        self.results["web_crawler"] = results
        
        return {
            "individual_results": results,
            "aggregate_metrics": aggregate_results,
            "total_test_cases": len(test_cases),
            "successful_evaluations": len([r for r in results if "error" not in r])
        }
    
    async def run_comprehensive_evaluation(self, test_suite: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all agents
        
        Args:
            test_suite: Dictionary containing test cases for each agent
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive evaluation...")
        
        evaluation_results = {}
        
        # Evaluate each agent (skip web_crawler as requested)
        # Each agent saves its results immediately, so we don't lose progress if one fails
        
        if "ad_performance" in test_suite:
            try:
                logger.info("Starting Ad Performance Agent evaluation...")
                evaluation_results["ad_performance"] = await self.evaluate_ad_performance_agent(
                    test_suite["ad_performance"]
                )
                logger.info("✅ Ad Performance Agent evaluation completed successfully")
            except Exception as e:
                logger.error(f"❌ Ad Performance Agent evaluation failed: {e}")
                evaluation_results["ad_performance"] = {"error": str(e)}
        
        if "blog_search" in test_suite:
            try:
                logger.info("Starting Blog Search Agent evaluation...")
                evaluation_results["blog_search"] = await self.evaluate_blog_search_agent(
                    test_suite["blog_search"]
                )
                logger.info("✅ Blog Search Agent evaluation completed successfully")
            except Exception as e:
                logger.error(f"❌ Blog Search Agent evaluation failed: {e}")
                evaluation_results["blog_search"] = {"error": str(e)}
        
        if "ad_rewriter" in test_suite:
            try:
                logger.info("Starting Ad Rewriter Agent evaluation...")
                evaluation_results["ad_rewriter"] = await self.evaluate_ad_rewriter_agent(
                    test_suite["ad_rewriter"]
                )
                logger.info("✅ Ad Rewriter Agent evaluation completed successfully")
            except Exception as e:
                logger.error(f"❌ Ad Rewriter Agent evaluation failed: {e}")
                evaluation_results["ad_rewriter"] = {"error": str(e)}
        
        # web_crawler evaluation skipped as requested
        
        # Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report(evaluation_results)
        
        # Save final comprehensive results (individual agent results already saved)
        try:
            await self.save_results(evaluation_results, comprehensive_report)
            logger.info("✅ Final comprehensive results saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save final comprehensive results: {e}")
            logger.info("⚠️  Individual agent results were already saved during evaluation")
        
        return {
            "evaluation_results": evaluation_results,
            "comprehensive_report": comprehensive_report,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict], agent_type: str) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results"""
        if not results or all("error" in r for r in results):
            return {"error": "No successful evaluations"}
        
        successful_results = [r for r in results if "error" not in r]
        
        aggregate = {
            "total_cases": len(results),
            "successful_cases": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0
        }
        
        # Calculate agent-specific aggregates
        if agent_type == "blog_search":
            relevance_scores = [r.get("relevance_scores", {}).get("query_response_relevance", 0) 
                             for r in successful_results]
            if relevance_scores:
                aggregate["avg_relevance_score"] = sum(relevance_scores) / len(relevance_scores)
                aggregate["min_relevance_score"] = min(relevance_scores)
                aggregate["max_relevance_score"] = max(relevance_scores)
        
        elif agent_type == "ad_rewriter":
            bleu_scores = [r.get("bleu_score", 0) for r in successful_results if "bleu_score" in r]
            if bleu_scores:
                aggregate["avg_bleu_score"] = sum(bleu_scores) / len(bleu_scores)
        
        return aggregate
    
    def _generate_comprehensive_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report"""
        report = {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "agents_evaluated": list(evaluation_results.keys()),
                "total_test_cases": sum(r.get("total_test_cases", 0) for r in evaluation_results.values()),
                "overall_success_rate": 0
            },
            "agent_performance": {},
            "recommendations": []
        }
        
        # Calculate overall success rate
        total_successful = sum(r.get("successful_evaluations", 0) for r in evaluation_results.values())
        total_cases = sum(r.get("total_test_cases", 0) for r in evaluation_results.values())
        
        if total_cases > 0:
            report["evaluation_summary"]["overall_success_rate"] = total_successful / total_cases
        
        # Analyze each agent's performance
        for agent_name, results in evaluation_results.items():
            aggregate_metrics = results.get("aggregate_metrics", {})
            success_rate = aggregate_metrics.get("success_rate", 0)
            
            performance_level = "excellent" if success_rate >= 0.9 else \
                              "good" if success_rate >= 0.7 else \
                              "needs_improvement" if success_rate >= 0.5 else "poor"
            
            report["agent_performance"][agent_name] = {
                "performance_level": performance_level,
                "success_rate": success_rate,
                "key_metrics": aggregate_metrics
            }
            
            # Generate recommendations
            if success_rate < 0.7:
                report["recommendations"].append(f"{agent_name}: Consider improving model parameters or training data")
        
        return report
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert Pydantic models and other objects
            return self._make_json_serializable(obj.__dict__)
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            # Convert enums, other types to string
            return str(obj)
    
    async def _save_agent_results(self, agent_name: str, results: Dict[str, Any]):
        """Save individual agent results immediately after evaluation"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert to JSON serializable format
            serializable_results = self._make_json_serializable(results)
            
            # Save individual agent results
            agent_file = self.output_dir / f"{agent_name}_results_{timestamp}.json"
            with open(agent_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved {agent_name} results to {agent_file}")
            
            # Also save a progress report
            completed_agents = list(self.results.keys())
            progress_report = {
                "timestamp": datetime.now().isoformat(),
                "completed_agents": completed_agents,
                "total_agents_planned": 3,  # ad_performance, blog_search, ad_rewriter
                "progress": f"{len(completed_agents)}/3 agents completed",
                "latest_results": {
                    agent_name: {
                        "total_test_cases": results.get("total_test_cases", 0),
                        "successful_evaluations": results.get("successful_evaluations", 0),
                        "success_rate": results.get("aggregate_metrics", {}).get("success_rate", 0)
                    }
                }
            }
            
            progress_file = self.output_dir / f"progress_report_{timestamp}.json"
            with open(progress_file, 'w') as f:
                json.dump(progress_report, f, indent=2)
            
            logger.info(f"Saved progress report to {progress_file}")
            
        except Exception as e:
            logger.error(f"Failed to save {agent_name} results: {e}")
            # Don't raise the exception, just log it so evaluation can continue

    async def save_results(self, evaluation_results: Dict[str, Any], comprehensive_report: Dict[str, Any]):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to JSON serializable format
        serializable_results = self._make_json_serializable(evaluation_results)
        serializable_report = self._make_json_serializable(comprehensive_report)
        
        # Save detailed results
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save comprehensive report
        report_file = self.output_dir / f"evaluation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        # Save CSV summary for easy analysis
        self._save_csv_summary(evaluation_results, timestamp)
        
        logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def _save_csv_summary(self, evaluation_results: Dict[str, Any], timestamp: str):
        """Save a CSV summary of results"""
        summary_data = []
        
        for agent_name, results in evaluation_results.items():
            for result in results.get("individual_results", []):
                if "error" not in result:
                    row = {
                        "agent": agent_name,
                        "test_case_id": result.get("test_case_id", ""),
                        "timestamp": result.get("timestamp", ""),
                    }
                    
                    # Add agent-specific metrics
                    if agent_name == "blog_search" and "relevance_scores" in result:
                        row["relevance_score"] = result["relevance_scores"].get("query_response_relevance", 0)
                    elif agent_name == "ad_rewriter" and "bleu_score" in result:
                        row["bleu_score"] = result.get("bleu_score", 0)
                    
                    summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.output_dir / f"evaluation_summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False) 