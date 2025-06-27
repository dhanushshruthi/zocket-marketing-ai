#!/usr/bin/env python3
"""
Marketing AI Agents Evaluation Runner

This script runs comprehensive evaluations of all marketing AI agents
and generates detailed reports with metrics like relevance, hallucination rate,
F1 scores, and ROUGE scores.

Usage:
    python evaluation/run_evaluation.py [--test-suite custom_test_suite.json] [--output-dir results/]
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
import json

# Add the parent directory to the path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.evaluator import MarketingAgentEvaluator
from evaluation.test_data import TestDataGenerator, BenchmarkDatasets
from evaluation.metrics import EvaluationMetrics

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )

async def run_benchmark_tests():
    """Run benchmark tests for individual metrics validation"""
    logger = logging.getLogger(__name__)
    logger.info("Running benchmark validation tests...")
    
    metrics = EvaluationMetrics()
    benchmark_results = {}
    
    # Test relevance scoring
    relevance_benchmark = BenchmarkDatasets.get_marketing_relevance_benchmark()
    relevance_results = []
    
    for test_case in relevance_benchmark:
        relevant_score = metrics.calculate_relevance_score(
            query=test_case["query"],
            response=test_case["relevant_response"]
        )
        
        irrelevant_score = metrics.calculate_relevance_score(
            query=test_case["query"],
            response=test_case["irrelevant_response"]
        )
        
        relevance_results.append({
            "query": test_case["query"],
            "relevant_score": relevant_score.get("query_response_relevance", 0),
            "irrelevant_score": irrelevant_score.get("query_response_relevance", 0),
            "expected_score": test_case["expected_relevance_score"]
        })
    
    benchmark_results["relevance"] = relevance_results
    
    # Test hallucination detection
    hallucination_benchmark = BenchmarkDatasets.get_hallucination_detection_benchmark()
    hallucination_results = []
    
    for test_case in hallucination_benchmark:
        result = metrics.detect_hallucination(
            response=test_case["response"],
            ground_truth=test_case["ground_truth"]
        )
        
        hallucination_results.append({
            "response": test_case["response"],
            "detected_rate": result.get("hallucination_rate", 0),
            "expected_rate": test_case["expected_hallucination_rate"]
        })
    
    benchmark_results["hallucination"] = hallucination_results
    
    # Test extraction F1
    extraction_benchmark = BenchmarkDatasets.get_extraction_benchmark()
    extraction_results = []
    
    for test_case in extraction_benchmark:
        result = metrics.calculate_extraction_f1(
            predicted_entities=test_case["predicted_entities"],
            true_entities=test_case["true_entities"]
        )
        
        extraction_results.append({
            "predicted": test_case["predicted_entities"],
            "true": test_case["true_entities"],
            "calculated_f1": result.get("f1_score", 0),
            "expected_f1": test_case["expected_f1_score"]
        })
    
    benchmark_results["extraction"] = extraction_results
    
    logger.info("Benchmark validation completed")
    return benchmark_results

async def main():
    """Main evaluation runner"""
    parser = argparse.ArgumentParser(description="Run Marketing AI Agents Evaluation")
    parser.add_argument("--test-suite", type=str, help="Path to custom test suite JSON file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                        help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--benchmark-only", action="store_true", 
                        help="Run only benchmark validation tests")
    parser.add_argument("--generate-test-suite", action="store_true",
                        help="Generate and save a new test suite")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Marketing AI Agents Evaluation")
    
    # Initialize evaluator
    evaluator = MarketingAgentEvaluator(output_dir=args.output_dir)
    
    try:
        # Run benchmark tests if requested
        if args.benchmark_only:
            benchmark_results = await run_benchmark_tests()
            
            # Save benchmark results
            output_path = Path(args.output_dir)
            output_path.mkdir(exist_ok=True)
            
            with open(output_path / "benchmark_results.json", 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            logger.info(f"Benchmark results saved to {output_path / 'benchmark_results.json'}")
            return
        
        # Generate test suite if requested
        if args.generate_test_suite:
            logger.info("Generating new test suite...")
            generator = TestDataGenerator()
            test_suite = generator.generate_comprehensive_test_suite()
            
            output_path = Path(args.output_dir)
            output_path.mkdir(exist_ok=True)
            suite_path = output_path / "generated_test_suite.json"
            
            generator.save_test_suite(test_suite, str(suite_path))
            logger.info(f"Test suite generated and saved to {suite_path}")
            
            if not args.test_suite:
                args.test_suite = str(suite_path)
        
        # Load test suite
        if args.test_suite:
            logger.info(f"Loading test suite from {args.test_suite}")
            generator = TestDataGenerator()
            test_suite = generator.load_test_suite(args.test_suite)
        else:
            logger.info("Generating default test suite...")
            generator = TestDataGenerator()
            test_suite = generator.generate_comprehensive_test_suite()
        
        # Run comprehensive evaluation
        logger.info("Starting comprehensive evaluation...")
        results = await evaluator.run_comprehensive_evaluation(test_suite)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        comprehensive_report = results["comprehensive_report"]
        summary = comprehensive_report["evaluation_summary"]
        
        print(f"Total Test Cases: {summary['total_test_cases']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.2%}")
        print(f"Agents Evaluated: {', '.join(summary['agents_evaluated'])}")
        
        print(f"\nAgent Performance:")
        for agent_name, performance in comprehensive_report["agent_performance"].items():
            print(f"  {agent_name}: {performance['performance_level']} "
                  f"({performance['success_rate']:.2%} success rate)")
        
        if comprehensive_report["recommendations"]:
            print(f"\nRecommendations:")
            for rec in comprehensive_report["recommendations"]:
                print(f"  • {rec}")
        
        print(f"\nDetailed results saved to: {evaluator.output_dir}")
        print("="*60)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 