# Marketing AI Agents Evaluation Framework

This evaluation framework provides comprehensive testing and benchmarking capabilities for all marketing AI agents, including metrics for relevance, hallucination detection, F1 scores for extraction tasks, and ROUGE scores for summaries.

## Features

- **Relevance Scoring**: Semantic similarity-based relevance evaluation using sentence transformers
- **Hallucination Detection**: Identifies unsupported claims in AI responses
- **F1 Score Calculation**: Precision, recall, and F1 scores for extraction tasks
- **ROUGE Scoring**: Summary quality evaluation with ROUGE-1, ROUGE-2, and ROUGE-L
- **Content Quality Analysis**: Marketing-specific content evaluation
- **Automated Test Generation**: Generates realistic test cases for all agents
- **Comprehensive Reporting**: Detailed reports with visualizations and recommendations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Basic Evaluation

```bash
# Run evaluation with auto-generated test cases
python evaluation/run_evaluation.py

# Generate and save test suite for later use
python evaluation/run_evaluation.py --generate-test-suite

# Run only benchmark validation tests
python evaluation/run_evaluation.py --benchmark-only
```

### 3. View Results

Results are saved in the `evaluation_results/` directory:
- `evaluation_results_[timestamp].json` - Detailed results for all test cases
- `evaluation_report_[timestamp].json` - Comprehensive summary report
- `evaluation_summary_[timestamp].csv` - CSV summary for analysis

## Evaluation Metrics

### 1. Relevance Scoring
- **Query-Response Relevance**: Cosine similarity between query and response embeddings
- **Context Relevance**: How well the response aligns with provided context
- **Categories**: highly_relevant (≥0.8), relevant (≥0.6), somewhat_relevant (≥0.4), not_relevant (<0.4)

### 2. Hallucination Detection
- **Sentence-Level Analysis**: Each sentence checked against ground truth
- **Support Scoring**: Similarity threshold for factual support (default: 0.3)
- **Confidence Levels**: high (≤10% hallucination), medium (≤30%), low (>30%)

### 3. F1 Score for Extraction
- **Entity Extraction**: Precision, recall, F1 for extracted entities
- **Document Retrieval**: Evaluation of retrieved document relevance
- **Campaign Analysis**: Accuracy of identified top performers

### 4. ROUGE Scores for Summaries
- **ROUGE-1**: Unigram overlap between reference and generated summary
- **ROUGE-2**: Bigram overlap evaluation
- **ROUGE-L**: Longest common subsequence evaluation

### 5. Content Quality Metrics
- **Readability Score**: Based on sentence complexity and word difficulty
- **Marketing Indicators**: Presence of CTAs, urgency indicators
- **Platform Compliance**: Character limits and best practices

## Agent-Specific Evaluations

### Ad Performance Analyzer
- **Metrics Accuracy**: Percentage error for CTR, CPA, ROAS predictions
- **Insight Quality**: ROUGE scores for generated insights
- **Recommendation Relevance**: F1 scores for top performer identification

### Blog Search Agent  
- **Search Relevance**: Query-response semantic similarity
- **Document Retrieval**: F1 scores for retrieved documents
- **Hallucination Detection**: Factual accuracy of responses

### Ad Text Rewriter
- **Semantic Preservation**: Similarity with original content
- **Tone Accuracy**: Consistency with requested tone
- **Platform Optimization**: Compliance with platform guidelines
- **Content Quality**: Readability and marketing effectiveness

### Web Crawler
- **Content Extraction**: Accuracy of title and content extraction
- **Keyword Extraction**: F1 scores for identified keywords
- **Content Quality**: Assessment of extracted content usefulness

## Usage Examples

### Running Custom Evaluations

```python
import asyncio
from evaluation.evaluator import MarketingAgentEvaluator
from evaluation.test_data import TestDataGenerator

async def run_custom_evaluation():
    # Initialize evaluator
    evaluator = MarketingAgentEvaluator(output_dir="my_results")
    
    # Generate custom test cases
    generator = TestDataGenerator()
    test_cases = generator.generate_blog_search_test_cases(10)
    
    # Run evaluation
    results = await evaluator.evaluate_blog_search_agent(test_cases)
    
    print(f"Success rate: {results['aggregate_metrics']['success_rate']:.2%}")

asyncio.run(run_custom_evaluation())
```

### Using Individual Metrics

```python
from evaluation.metrics import EvaluationMetrics

metrics = EvaluationMetrics()

# Calculate relevance score
relevance = metrics.calculate_relevance_score(
    query="Facebook advertising best practices",
    response="Use targeted audiences and compelling visuals for better Facebook ad performance."
)

print(f"Relevance: {relevance['query_response_relevance']:.3f}")

# Detect hallucinations
hallucination = metrics.detect_hallucination(
    response="Facebook ads have a 50% click-through rate on average.",
    ground_truth=["Facebook ads typically have 1-2% click-through rates"]
)

print(f"Hallucination rate: {hallucination['hallucination_rate']:.2%}")

# Calculate F1 for extraction
f1_results = metrics.calculate_extraction_f1(
    predicted_entities=["Facebook", "Instagram", "Google"],
    true_entities=["Facebook", "Instagram", "Twitter"]
)

print(f"F1 Score: {f1_results['f1_score']:.3f}")
```

### Custom Test Data

```python
from evaluation.test_data import TestDataGenerator

generator = TestDataGenerator(seed=123)

# Generate custom test cases
custom_tests = {
    "ad_performance": generator.generate_ad_performance_test_cases(20),
    "blog_search": generator.generate_blog_search_test_cases(15)
}

# Save for later use
generator.save_test_suite(custom_tests, "my_test_suite.json")
```

## Configuration Options

### Command Line Arguments

```bash
python evaluation/run_evaluation.py \
    --test-suite custom_tests.json \
    --output-dir results/ \
    --log-level DEBUG \
    --generate-test-suite \
    --benchmark-only
```

### Environment Variables

```bash
# Optional: Configure evaluation parameters
export EVALUATION_SIMILARITY_THRESHOLD=0.3
export EVALUATION_OUTPUT_DIR=evaluation_results
export EVALUATION_LOG_LEVEL=INFO
```

## Benchmarking

The framework includes predefined benchmark datasets for validating metric accuracy:

```bash
# Run benchmark validation
python evaluation/run_evaluation.py --benchmark-only
```

Benchmark results validate:
- Relevance scoring accuracy against known relevant/irrelevant pairs
- Hallucination detection sensitivity 
- F1 score calculation correctness

## Output Analysis

### JSON Results Structure

```json
{
  "evaluation_results": {
    "ad_performance": {
      "individual_results": [...],
      "aggregate_metrics": {
        "total_cases": 25,
        "successful_cases": 23,
        "success_rate": 0.92
      }
    }
  },
  "comprehensive_report": {
    "evaluation_summary": {
      "overall_success_rate": 0.87,
      "agents_evaluated": ["ad_performance", "blog_search"]
    },
    "recommendations": [...]
  }
}
```

### CSV Analysis

Import the generated CSV files into Excel, pandas, or other analysis tools:

```python
import pandas as pd

# Load evaluation summary
df = pd.read_csv('evaluation_results/evaluation_summary_20241127_143022.csv')

# Analyze by agent
agent_performance = df.groupby('agent')['relevance_score'].mean()
print(agent_performance)
```

## Troubleshooting

### Common Issues

1. **ImportError for sentence_transformers**
   ```bash
   pip install sentence-transformers
   ```

2. **NLTK data not found**
   ```python
   import nltk
   nltk.download('punkt')
   ```

3. **Memory issues with large test suites**
   - Reduce test case numbers in `TestDataGenerator`
   - Process agents individually

### Debug Mode

```bash
python evaluation/run_evaluation.py --log-level DEBUG
```

This provides detailed logging for troubleshooting evaluation issues.

## Contributing

To add new evaluation metrics:

1. Add metric methods to `evaluation/metrics.py`
2. Update agent evaluators in `evaluation/evaluator.py`  
3. Add test cases to `evaluation/test_data.py`
4. Update documentation

## License

This evaluation framework is part of the Marketing AI Agents project and follows the same MIT license terms. 