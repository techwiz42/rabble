# rabble/adapters/factory.py
from typing import Optional, Dict, Any

from .base import ModelAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .deepseek_adapter import DeepSeekAdapter

class ModelAdapterFactory:
    """Factory class for creating model adapters."""
    
    # Default models for each provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20240620",
        "deepseek": "deepseek-chat"
    }
    
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
            provider: Provider name (openai, anthropic, deepseek)
            client: Optional client instance
            model: Optional model override
            **kwargs: Additional provider-specific initialization parameters
            
        Returns:
            A ModelAdapter instance
        """
        provider = provider.lower()
        
        if provider not in ModelAdapterFactory.DEFAULT_MODELS:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Use default model if none specified
        if not model:
            model = ModelAdapterFactory.DEFAULT_MODELS[provider]
        
        # Create appropriate adapter
        if provider == "openai":
            return OpenAIAdapter(client=client, default_model=model, **kwargs)
        elif provider == "anthropic":
            return AnthropicAdapter(client=client, default_model=model, **kwargs)
        elif provider == "deepseek":
            return DeepSeekAdapter(client=client, default_model=model, **kwargs)
