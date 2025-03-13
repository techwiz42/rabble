# tests/test_api_keys.py
import os
import pytest
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add rabble directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rabble.adapters import (
    OpenAIAdapter, 
    AnthropicAdapter, 
    DeepSeekAdapter, 
    MistralAdapter, 
    CohereAdapter, 
    GoogleAdapter 
)

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / "rabble" / ".env")

def test_openai_key():
    """Test if the OpenAI API key is valid."""
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY not found in environment variables"
    
    # Get model from environment
    model = os.getenv("OPENAI_DEFAULT_MODEL")
    assert model, "OPENAI_DEFAULT_MODEL not found in environment variables"
    
    # Create the adapter
    adapter = OpenAIAdapter(default_model=model)
    
    # Simple test message
    messages = [{"role": "user", "content": "Hello, this is a test message."}]
    
    try:
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10  # Minimal response to save tokens
        )
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"OpenAI test passed ({model}): {response.get('content')}")
        return True
    except Exception as e:
        print(f"OpenAI test failed: {str(e)}")
        return False

def test_anthropic_key():
    """Test if the Anthropic API key is valid."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    assert api_key, "ANTHROPIC_API_KEY not found in environment variables"
    
    # Get model from environment
    model = os.getenv("ANTHROPIC_DEFAULT_MODEL")
    assert model, "ANTHROPIC_DEFAULT_MODEL not found in environment variables"
    
    # Create the adapter
    adapter = AnthropicAdapter(default_model=model)
    
    # Simple test message
    messages = [{"role": "user", "content": "Hello, this is a test message."}]
    
    try:
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10  # Minimal response to save tokens
        )
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"Anthropic test passed ({model}): {response.get('content')}")
        return True
    except Exception as e:
        print(f"Anthropic test failed: {str(e)}")
        return False

def test_deepseek_key():
    """Test if the DeepSeek API key is valid."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    assert api_key, "DEEPSEEK_API_KEY not found in environment variables"
    
    # Get model from environment
    model = os.getenv("DEEPSEEK_DEFAULT_MODEL")
    assert model, "DEEPSEEK_DEFAULT_MODEL not found in environment variables"
    
    try:
        # Import DeepSeekAPI and set up client
        from deepseek import DeepSeekAPI
        client = DeepSeekAPI()
        
        # Create the adapter
        adapter = DeepSeekAdapter(client=client, default_model=model)
        
        # Simple test message
        messages = [{"role": "user", "content": "Hello, this is a test message."}]
        
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10  # Minimal response to save tokens
        )
        
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"DeepSeek test passed ({model}): {response.get('content')}")
        return True
    except Exception as e:
        print(f"DeepSeek test failed: {str(e)}")
        return False

def test_mistral_key():
    """Test if the Mistral AI API key is valid."""
    api_key = os.getenv("MISTRAL_API_KEY")
    assert api_key, "MISTRAL_API_KEY not found in environment variables"
    
    # Get model from environment
    model = os.getenv("MISTRAL_DEFAULT_MODEL")
    assert model, "MISTRAL_DEFAULT_MODEL not found in environment variables"
    
    try:
        # Try with new client first
        try:
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            new_client = True
            print("Using Mistral v1.x client")
        except ImportError:
            # Fall back to old client
            from mistralai.client import MistralClient
            client = MistralClient(api_key=api_key)
            new_client = False
            print("Using Mistral v0.x client")
        
        # Create the adapter with the appropriate client
        adapter = MistralAdapter(client=client, api_key=api_key, default_model=model)
        
        # Simple test message
        messages = [{"role": "user", "content": "Hello, this is a test message."}]
        
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10  # Minimal response to save tokens
        )
        
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"Mistral test passed ({model}): {response.get('content')}")
        return True
    except ImportError as e:
        print(f"Mistral test failed - missing dependency: {str(e)}")
        return False
    except Exception as e:
        print(f"Mistral test failed: {str(e)}")
        return False

def test_cohere_key():
    """Test if the Cohere API key is valid."""
    api_key = os.getenv("COHERE_API_KEY")
    assert api_key, "COHERE_API_KEY not found in environment variables"
    
    # Get model from environment
    model = os.getenv("COHERE_DEFAULT_MODEL")
    assert model, "COHERE_DEFAULT_MODEL not found in environment variables"
    
    try:
        # Import Cohere client
        import cohere
        client = cohere.Client(api_key=api_key)
        
        # Create the adapter
        adapter = CohereAdapter(client=client, api_key=api_key, default_model=model)
        
        # Simple test message
        messages = [{"role": "user", "content": "Hello, this is a test message."}]
        
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10  # Minimal response to save tokens
        )
        
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"Cohere test passed ({model}): {response.get('content')}")
        return True
    except Exception as e:
        print(f"Cohere test failed: {str(e)}")
        return False

def test_google_key():
    """Test if the Google API key is valid."""
    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key, "GOOGLE_API_KEY not found in environment variables"
    
    # Get model from environment - note the typo fix from DEFAUL -> DEFAULT
    model = os.getenv("GOOGLE_DEFAULT_MODEL")
    assert model, "GOOGLE_DEFAULT_MODEL not found in environment variables"
    
    try:
        # Import Google GenerativeAI
        import google.generativeai as genai
        
        # Create the adapter
        adapter = GoogleAdapter(api_key=api_key, default_model=model)
        
        # Simple test message
        messages = [{"role": "user", "content": "Hello, this is a test message."}]
        
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            model=model,
            max_tokens=10  # Minimal response to save tokens
        )
        
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"Google test passed ({model}): {response.get('content')}")
        return True
    except Exception as e:
        print(f"Google test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing API keys...")
    test_results = {
        "OpenAI": test_openai_key(),
        "Anthropic": test_anthropic_key(),
        "DeepSeek": test_deepseek_key(),
        "Mistral": test_mistral_key(),
        "Cohere": test_cohere_key(),
        "Google": test_google_key(),
    }
    
    print("\nSummary:")
    for provider, success in test_results.items():
        print(f"{provider} API key: {'✓ Valid' if success else '✗ Invalid'}")
