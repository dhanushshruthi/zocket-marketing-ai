"""
Marketing Blog Search Agent

This multi-step agent uses a vector database (ChromaDB) to search a set of marketing blogs
and answer user queries using Agentic RAG approach.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from app.models.schemas import BlogSearchRequest, BlogSearchResponse, BlogSearchResult
from app.database.chroma_client import ChromaClient
from app.utils.azure_openai_client import get_azure_openai_client

logger = logging.getLogger(__name__)


class BlogSearchAgent:
    """Multi-step agent for searching marketing blogs using vector database and RAG."""
    
    def __init__(self, chroma_client: ChromaClient):
        self.chroma_client = chroma_client
        self.azure_client = get_azure_openai_client()
        self.system_prompt = """
        You are an expert marketing researcher and content strategist specializing in digital marketing best practices.
        
        Your role is to:
        1. Analyze user queries about marketing topics
        2. Synthesize information from multiple marketing blog sources
        3. Provide comprehensive, actionable answers
        4. Reference specific insights from the retrieved content
        5. Suggest related topics and queries for further exploration
        
        Always provide practical, evidence-based marketing advice with clear reasoning.
        When citing information, mention the relevance and applicability to the user's query.
        """
    
    async def search_and_answer(self, request: BlogSearchRequest) -> BlogSearchResponse:
        """Perform multi-step search and generate comprehensive answer."""
        try:
            start_time = time.time()
            logger.info(f"Processing blog search query: {request.query}")
            
            initial_results = await self._perform_vector_search(request)
            
            expanded_queries = await self._analyze_and_expand_query(request.query)
            
            comprehensive_results = await self._perform_multi_query_search(
                expanded_queries, request.max_results
            )
            
            combined_results = self._combine_and_deduplicate_results(
                initial_results, comprehensive_results
            )
            
            enhanced_results = await self._enhance_results_with_ai(
                request.query, combined_results
            )
            
            suggested_queries = await self._generate_suggested_queries(request.query, enhanced_results)
            
            search_time = (time.time() - start_time) * 1000
            
            return BlogSearchResponse(
                query=request.query,
                results=enhanced_results,
                total_results=len(enhanced_results),
                search_time_ms=round(search_time, 2),
                suggested_queries=suggested_queries
            )
            
        except Exception as e:
            logger.error(f"Failed to process blog search: {e}")
            raise
    
    async def _perform_vector_search(self, request: BlogSearchRequest) -> List[Dict[str, Any]]:
        """Perform initial vector search in ChromaDB."""
        try:
            filter_metadata = {}
            if request.filter_category:
                filter_metadata["category"] = request.filter_category
            if request.filter_topic:
                filter_metadata["topic"] = request.filter_topic
            
            results = await self.chroma_client.search(
                query=request.query,
                n_results=request.max_results,
                filter_metadata=filter_metadata if filter_metadata else None
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform vector search: {e}")
            raise
    
    async def _analyze_and_expand_query(self, query: str) -> List[str]:
        """Analyze query and generate related search terms for comprehensive coverage."""
        try:
            prompt = f"""
            Analyze the following marketing query and generate 3-4 related search terms that would help provide comprehensive coverage of the topic:
            
            Original Query: "{query}"
            
            Generate related queries that cover:
            - Different aspects of the main topic
            - Related marketing concepts
            - Specific implementation strategies
            - Alternative terminology or approaches
            
            Return only the search terms, one per line, without numbering or bullets.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.3
            )
            
            expanded_queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith(('Original Query:', '-', '•', '1.', '2.', '3.', '4.')):
                    expanded_queries.append(line.strip('"'))
            
            return expanded_queries[:4]
            
        except Exception as e:
            logger.error(f"Failed to analyze and expand query: {e}")
            return []
    
    async def _perform_multi_query_search(self, queries: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Perform searches for multiple related queries."""
        try:
            all_results = []
            results_per_query = max(1, max_results // len(queries)) if queries else max_results
            
            for query in queries:
                try:
                    results = await self.chroma_client.search(
                        query=query,
                        n_results=results_per_query
                    )
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Failed to search for query '{query}': {e}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Failed to perform multi-query search: {e}")
            return []
    
    def _combine_and_deduplicate_results(
        self, 
        initial_results: List[Dict[str, Any]], 
        additional_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine results from different searches and remove duplicates."""
        try:
            seen_ids = set()
            combined_results = []
            
            for result in initial_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    combined_results.append(result)
            
            for result in additional_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    combined_results.append(result)
            
            combined_results.sort(key=lambda x: x['distance'])
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to combine and deduplicate results: {e}")
            return initial_results
    
    async def _enhance_results_with_ai(
        self, 
        original_query: str, 
        search_results: List[Dict[str, Any]]
    ) -> List[BlogSearchResult]:
        """Enhance search results with AI-generated context and relevance analysis."""
        try:
            enhanced_results = []
            
            for result in search_results:
                try:
                    relevance_explanation = await self._generate_relevance_explanation(
                        original_query, result['content']
                    )
                    
                    relevance_score = max(0, min(1, 1 - result['distance']))
                    
                    enhanced_result = BlogSearchResult(
                        id=result['id'],
                        content=result['content'],
                        metadata={
                            **result['metadata'],
                            'relevance_explanation': relevance_explanation
                        },
                        relevance_score=round(relevance_score, 3)
                    )
                    
                    enhanced_results.append(enhanced_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to enhance result {result['id']}: {e}")
                    enhanced_result = BlogSearchResult(
                        id=result['id'],
                        content=result['content'],
                        metadata=result['metadata'],
                        relevance_score=max(0, min(1, 1 - result['distance']))
                    )
                    enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to enhance results with AI: {e}")
            basic_results = []
            for result in search_results:
                basic_results.append(BlogSearchResult(
                    id=result['id'],
                    content=result['content'],
                    metadata=result['metadata'],
                    relevance_score=max(0, min(1, 1 - result['distance']))
                ))
            return basic_results
    
    async def _generate_relevance_explanation(self, query: str, content: str) -> str:
        """Generate AI explanation of why content is relevant to the query."""
        try:
            prompt = f"""
            Explain in 1-2 sentences why the following content is relevant to the user's query:
            
            User Query: "{query}"
            
            Content: "{content[:500]}..."  # Truncate for efficiency
            
            Focus on specific connections between the query and the content.
            Be concise and practical.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=100
            )
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate relevance explanation: {e}")
            return "Content related to your marketing query."
    
    async def _generate_suggested_queries(
        self, 
        original_query: str, 
        results: List[BlogSearchResult]
    ) -> List[str]:
        """Generate suggested related queries based on search results."""
        try:
            topics = set()
            categories = set()
            
            for result in results:
                if 'topic' in result.metadata:
                    topics.add(result.metadata['topic'])
                if 'category' in result.metadata:
                    categories.add(result.metadata['category'])
            
            context = f"""
            Original Query: "{original_query}"
            Related Topics Found: {', '.join(list(topics)[:5])}
            Related Categories: {', '.join(list(categories)[:3])}
            """
            
            prompt = f"""
            Based on the following search context, suggest 3-4 related marketing queries that the user might find helpful:
            
            {context}
            
            Generate specific, actionable queries that would help the user explore related marketing topics.
            Focus on practical questions that marketers commonly ask.
            
            Return only the suggested queries, one per line.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.azure_client.generate_completion(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.4,
                max_tokens=200
            )
            
            suggested_queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith(('Based on', 'Original Query:', '-', '•')):
                    suggested_queries.append(line.strip('"'))
            
            return suggested_queries[:4] 
            
        except Exception as e:
            logger.error(f"Failed to generate suggested queries: {e}")
            return [] 