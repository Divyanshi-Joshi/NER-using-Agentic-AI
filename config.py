"""
Azure OpenAI Configuration Module
Handles secure loading of Azure credentials from environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_azure_config():
    """
    Get Azure OpenAI configuration from environment variables.
    
    IMPORTANT: Set these environment variables before running the application:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
    - AZURE_DEPLOYMENT_NAME: Name of your chat model deployment
    - AZURE_EMBEDDING_DEPLOYMENT_NAME: Name of your embedding model deployment
    - AZURE_API_VERSION: API version for chat model (e.g., "2025-01-01-preview")
    - AZURE_EMBEDDING_API_VERSION: API version for embeddings (e.g., "2024-12-01-preview")
    
    Returns:
        dict: Configuration dictionary with Azure OpenAI credentials
        
    Raises:
        ValueError: If any required environment variable is missing
    """
    
    config = {
        # Chat Model configuration
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment_name": os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"),
        "api_version": os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
        
        # Embedding Model configuration (DIFFERENT API VERSION!)
        "embedding_deployment_name": os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
        "embedding_api_version": os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-12-01-preview")
    }
    
    return config


def verify_azure_config():
    """
    Verify all required Azure configuration variables are set.
    
    Raises:
        ValueError: If any required variable is missing
    """
    config = get_azure_config()
    required_keys = ["api_key", "azure_endpoint", "deployment_name"]
    
    for key in required_keys:
        if not config[key]:
            raise ValueError(
                f"Missing {key}. Please set the corresponding environment variable.\n"
                f"Required environment variables:\n"
                f"  - AZURE_OPENAI_API_KEY\n"
                f"  - AZURE_OPENAI_ENDPOINT\n"
                f"  - AZURE_DEPLOYMENT_NAME\n"
                f"  - AZURE_EMBEDDING_DEPLOYMENT_NAME\n"
                f"  - AZURE_API_VERSION\n"
                f"  - AZURE_EMBEDDING_API_VERSION\n"
                f"\nSee .env.example for reference."
            )
    
    print("✓ Azure OpenAI configuration verified")
    return config
