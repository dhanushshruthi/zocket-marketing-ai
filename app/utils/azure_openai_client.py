"""
Azure OpenAI client utility for marketing AI agents.
"""

import logging
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
import tiktoken

from app.utils.config import get_settings

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """Azure OpenAI client for generating responses."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.encoding = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Azure OpenAI client."""
        try:
            self.client = AzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint
            )
            
            # Initialize tokenizer for token counting
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
            logger.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            return 0
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate completion using Azure OpenAI."""
        try:
            # Prepare messages
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Use settings defaults if not provided
            temperature = temperature or self.settings.temperature
            max_tokens = max_tokens or self.settings.max_tokens
            
            # Generate completion
            response = self.client.chat.completions.create(
                model=self.settings.azure_openai_deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Azure OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.settings.azure_openai_embedding_deployment,
                input=texts
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise


# Global client instance
_azure_client = None


def get_azure_openai_client() -> AzureOpenAIClient:
    """Get Azure OpenAI client singleton."""
    global _azure_client
    if _azure_client is None:
        _azure_client = AzureOpenAIClient()
    return _azure_client 