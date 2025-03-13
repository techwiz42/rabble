# rabble/adapters/factory.py
from typing import Optional, Dict, Any
import os

from .base import ModelAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .mistral_adapter import MistralAdapter
from .cohere_adapter import CohereAdapter
from .google_adapter import GoogleAdapter

class ModelAdapterFactory:
    """Factory class for creating model adapters."""
    
    @staticmethod
    def create_adapter(
        provider: str,
        client: Optional[Any] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> ModelAdapter:
        """
        Create and return an adapter for the specified provider.
        
        Args:
            provider: Provider name (openai, anthropic, deepseek, mistral, cohere, google)
            client: Optional client instance
            model: Optional model override
            **kwargs: Additional provider-specific initialization parameters
            
        Returns:
            A ModelAdapter instance
            
        Raises:
            ValueError: If the provider is not supported or required environment variables are missing
        """
        provider = provider.lower()
        
        # Check if provider is supported
        if provider not in ["openai", "anthropic", "deepseek", "mistral", "cohere", "google"]:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Use provided model or get from environment
        if not model:
            env_var_name = f"{provider.upper()}_DEFAULT_MODEL"
            env_model = os.getenv(env_var_name)
            if not env_model:
                raise ValueError(f"No model specified and {env_var_name} environment variable not found in .env file")
            model = env_model
        
        # Create appropriate adapter
        if provider == "openai":
            return OpenAIAdapter(client=client, default_model=model, **kwargs)
        elif provider == "anthropic":
            return AnthropicAdapter(client=client, default_model=model, **kwargs)
        elif provider == "mistral":
            return MistralAdapter(client=client, default_model=model, **kwargs)
        elif provider == "cohere":
            return CohereAdapter(client=client, default_model=model, **kwargs)
        elif provider == "google":
            return GoogleAdapter(client=client, default_model=model, **kwargs)
