"""
Evaluation metrics for Marketing AI Agents

This module provides various metrics to evaluate the performance of AI agents:
- Relevance scoring using semantic similarity
- Hallucination detection
- F1 scores for extraction tasks
- ROUGE scores for summary evaluation
- Custom marketing-specific metrics
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Additional downloads for newer NLTK versions
try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    # Test tokenization to see if punkt works
    sent_tokenize("Test sentence.")
except Exception as e:
    try:
        nltk.download('punkt_tab')
        nltk.download('punkt')
    except:
        pass

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Comprehensive evaluation metrics for Marketing AI Agents"""
    
    def __init__(self):
        """Initialize evaluation metrics with required models"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except Exception as e:
            logger.error(f"Error initializing evaluation models: {e}")
            self.sentence_model = None
            self.rouge_scorer = None
    
    def calculate_relevance_score(self, query: str, response: str, context: List[str] = None) -> Dict[str, float]:
        """
        Calculate relevance score using semantic similarity
        
        Args:
            query: User query or question
            response: AI agent response
            context: Optional context documents for additional relevance checking
            
        Returns:
            Dictionary with relevance scores
        """
        if not self.sentence_model:
            return {"error": "Sentence model not initialized"}
        
        try:
            # Encode query and response
            query_embedding = self.sentence_model.encode([query])
            response_embedding = self.sentence_model.encode([response])
            
            # Calculate query-response relevance
            query_response_similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
            
            scores = {
                "query_response_relevance": float(query_response_similarity),
                "relevance_category": self._categorize_relevance(query_response_similarity)
            }
            
            # If context is provided, calculate context relevance
            if context:
                context_embeddings = self.sentence_model.encode(context)
                response_context_similarities = cosine_similarity(response_embedding, context_embeddings)[0]
                
                scores.update({
                    "max_context_relevance": float(np.max(response_context_similarities)),
                    "avg_context_relevance": float(np.mean(response_context_similarities)),
                    "context_support_score": float(np.mean([s for s in response_context_similarities if s > 0.3]))
                })
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return {"error": str(e)}
    
    def detect_hallucination(self, response: str, ground_truth: List[str], threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect potential hallucinations by checking if response content is supported by ground truth
        
        Args:
            response: AI agent response
            ground_truth: List of factual statements or source documents
            threshold: Minimum similarity threshold for factual support
            
        Returns:
            Dictionary with hallucination detection results
        """
        if not self.sentence_model or not ground_truth:
            return {"error": "Missing model or ground truth data"}
        
        try:
            # Split response into sentences with fallback
            try:
                response_sentences = nltk.sent_tokenize(response)
            except:
                # Fallback to simple sentence splitting
                response_sentences = [s.strip() for s in response.split('.') if s.strip()]
            
            # Encode all sentences and ground truth
            response_embeddings = self.sentence_model.encode(response_sentences)
            truth_embeddings = self.sentence_model.encode(ground_truth)
            
            # Check each sentence against ground truth
            unsupported_sentences = []
            sentence_scores = []
            
            for i, sentence in enumerate(response_sentences):
                similarities = cosine_similarity([response_embeddings[i]], truth_embeddings)[0]
                max_similarity = np.max(similarities)
                
                sentence_scores.append({
                    "sentence": sentence,
                    "max_support_score": float(max_similarity),
                    "is_supported": max_similarity >= threshold
                })
                
                if max_similarity < threshold:
                    unsupported_sentences.append(sentence)
            
            hallucination_rate = len(unsupported_sentences) / len(response_sentences) if response_sentences else 0
            
            return {
                "hallucination_rate": hallucination_rate,
                "unsupported_sentences": unsupported_sentences,
                "sentence_scores": sentence_scores,
                "total_sentences": len(response_sentences),
                "supported_sentences": len(response_sentences) - len(unsupported_sentences),
                "confidence_level": self._categorize_confidence(hallucination_rate)
            }
            
        except Exception as e:
            logger.error(f"Error detecting hallucination: {e}")
            return {"error": str(e)}
    
    def calculate_extraction_f1(self, predicted_entities: List[str], true_entities: List[str]) -> Dict[str, float]:
        """
        Calculate F1, precision, and recall scores for entity extraction tasks
        
        Args:
            predicted_entities: List of entities extracted by the model
            true_entities: List of ground truth entities
            
        Returns:
            Dictionary with extraction metrics
        """
        try:
            # Normalize entities (lowercase and strip)
            predicted_normalized = [entity.lower().strip() for entity in predicted_entities]
            true_normalized = [entity.lower().strip() for entity in true_entities]
            
            # Get unique entities
            all_entities = list(set(predicted_normalized + true_normalized))
            
            # Create binary vectors
            y_true = [1 if entity in true_normalized else 0 for entity in all_entities]
            y_pred = [1 if entity in predicted_normalized else 0 for entity in all_entities]
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Additional metrics
            true_positives = len(set(predicted_normalized) & set(true_normalized))
            false_positives = len(set(predicted_normalized) - set(true_normalized))
            false_negatives = len(set(true_normalized) - set(predicted_normalized))
            
            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "accuracy": float((len(all_entities) - false_positives - false_negatives) / len(all_entities)) if all_entities else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating extraction F1: {e}")
            return {"error": str(e)}
    
    def calculate_rouge_scores(self, reference: str, summary: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores for summary evaluation
        
        Args:
            reference: Reference/ground truth summary
            summary: Generated summary
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not self.rouge_scorer:
            return {"error": "ROUGE scorer not initialized"}
        
        try:
            scores = self.rouge_scorer.score(reference, summary)
            
            return {
                "rouge1_precision": scores['rouge1'].precision,
                "rouge1_recall": scores['rouge1'].recall,
                "rouge1_fmeasure": scores['rouge1'].fmeasure,
                "rouge2_precision": scores['rouge2'].precision,
                "rouge2_recall": scores['rouge2'].recall,
                "rouge2_fmeasure": scores['rouge2'].fmeasure,
                "rougeL_precision": scores['rougeL'].precision,
                "rougeL_recall": scores['rougeL'].recall,
                "rougeL_fmeasure": scores['rougeL'].fmeasure,
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {"error": str(e)}
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score for text generation evaluation
        
        Args:
            reference: Reference text
            candidate: Generated text
            
        Returns:
            BLEU score
        """
        try:
            # Try NLTK tokenization with fallback
            try:
                reference_tokens = nltk.word_tokenize(reference.lower())
                candidate_tokens = nltk.word_tokenize(candidate.lower())
            except:
                # Fallback to simple word splitting
                reference_tokens = reference.lower().split()
                candidate_tokens = candidate.lower().split()
            
            return sentence_bleu([reference_tokens], candidate_tokens)
            
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def evaluate_ad_performance_insights(self, predicted_insights: Dict[str, Any], 
                                       ground_truth_insights: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate ad performance analysis insights
        
        Args:
            predicted_insights: AI-generated insights
            ground_truth_insights: Expected insights
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            metrics = {}
            
            # Evaluate numerical predictions (CTR, CPA, etc.)
            for metric_name in ['ctr', 'cpa', 'roas', 'conversion_rate']:
                if metric_name in predicted_insights and metric_name in ground_truth_insights:
                    pred_value = predicted_insights[metric_name]
                    true_value = ground_truth_insights[metric_name]
                    
                    # Calculate percentage error
                    if true_value != 0:
                        percentage_error = abs(pred_value - true_value) / abs(true_value) * 100
                        metrics[f"{metric_name}_percentage_error"] = percentage_error
                        metrics[f"{metric_name}_accuracy"] = max(0, 100 - percentage_error) / 100
            
            # Evaluate categorical predictions (top performers, recommendations)
            if 'top_performers' in predicted_insights and 'top_performers' in ground_truth_insights:
                pred_performers = set(predicted_insights['top_performers'])
                true_performers = set(ground_truth_insights['top_performers'])
                
                f1_metrics = self.calculate_extraction_f1(list(pred_performers), list(true_performers))
                metrics['top_performers_f1'] = f1_metrics['f1_score']
                metrics['top_performers_precision'] = f1_metrics['precision']
                metrics['top_performers_recall'] = f1_metrics['recall']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ad performance insights: {e}")
            return {"error": str(e)}
    
    def evaluate_content_quality(self, content: str) -> Dict[str, Any]:
        """
        Evaluate content quality using various linguistic metrics
        
        Args:
            content: Text content to evaluate
            
        Returns:
            Dictionary with content quality metrics
        """
        try:
            # Basic metrics
            word_count = len(content.split())
            
            # Try NLTK sentence tokenization, fallback to simple split if fails
            try:
                sentence_count = len(nltk.sent_tokenize(content))
            except:
                # Fallback to simple sentence counting
                sentence_count = len([s for s in content.split('.') if s.strip()])
                
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Readability metrics (simplified)
            complex_words = len([word for word in content.split() if len(word) > 6])
            readability_score = max(0, 100 - (complex_words / word_count * 100)) if word_count > 0 else 0
            
            # Marketing-specific checks
            has_cta = bool(re.search(r'\b(buy|purchase|shop|click|learn more|sign up|get started)\b', content.lower()))
            has_urgency = bool(re.search(r'\b(now|today|limited|urgent|hurry|don\'t miss)\b', content.lower()))
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length,
                "readability_score": readability_score,
                "has_call_to_action": has_cta,
                "has_urgency_indicators": has_urgency,
                "complexity_ratio": complex_words / word_count if word_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error evaluating content quality: {e}")
            return {"error": str(e)}
    
    def _categorize_relevance(self, score: float) -> str:
        """Categorize relevance score into human-readable labels"""
        if score >= 0.8:
            return "highly_relevant"
        elif score >= 0.6:
            return "relevant"
        elif score >= 0.4:
            return "somewhat_relevant"
        else:
            return "not_relevant"
    
    def _categorize_confidence(self, hallucination_rate: float) -> str:
        """Categorize confidence based on hallucination rate"""
        if hallucination_rate <= 0.1:
            return "high_confidence"
        elif hallucination_rate <= 0.3:
            return "medium_confidence"
        else:
            return "low_confidence" 