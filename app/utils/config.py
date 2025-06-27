"""
Configuration management for Marketing AI Agents application.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(
        default="https://your-resource-name.openai.azure.com/",
        env="AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_api_key: str = Field(
        default="your-api-key-here",
        env="AZURE_OPENAI_API_KEY"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview",
        env="AZURE_OPENAI_API_VERSION"
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4",
        env="AZURE_OPENAI_DEPLOYMENT_NAME"
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-ada-002",
        env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
    
    # ChromaDB Configuration
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        env="CHROMA_DB_PATH"
    )
    chroma_collection_name: str = Field(
        default="marketing_knowledge",
        env="CHROMA_COLLECTION_NAME"
    )
    
    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        env="API_HOST"
    )
    api_port: int = Field(
        default=8000,
        env="API_PORT"
    )
    api_reload: bool = Field(
        default=True,
        env="API_RELOAD"
    )
    
    # Knowledge Graph Configuration
    enable_knowledge_graph: bool = Field(
        default=True,
        env="ENABLE_KNOWLEDGE_GRAPH"
    )
    graph_db_path: str = Field(
        default="./data/knowledge_graph",
        env="GRAPH_DB_PATH"
    )
    
    # Agent Configuration
    max_retries: int = Field(
        default=3,
        env="MAX_RETRIES"
    )
    temperature: float = Field(
        default=0.7,
        env="TEMPERATURE"
    )
    max_tokens: int = Field(
        default=2000,
        env="MAX_TOKENS"
    )
    
    # Evaluation Configuration
    evaluation_enabled: bool = Field(
        default=True,
        env="EVALUATION_ENABLED"
    )
    metrics_output_dir: str = Field(
        default="./data/metrics",
        env="METRICS_OUTPUT_DIR"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings 