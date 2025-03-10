# tests/test_api_keys.py
import os
import pytest
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add rabble directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rabble.adapters import OpenAIAdapter, AnthropicAdapter, DeepSeekAdapter

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / "rabble" / ".env")

def test_openai_key():
    """Test if the OpenAI API key is valid."""
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY not found in environment variables"
    
    # Create the adapter
    adapter = OpenAIAdapter()
    
    # Simple test message
    messages = [{"role": "user", "content": "Hello, this is a test message."}]
    
    try:
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            model="gpt-3.5-turbo",  # Using a smaller model for cost efficiency
            max_tokens=10  # Minimal response to save tokens
        )
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"OpenAI test passed: {response.get('content')}")
        return True
    except Exception as e:
        print(f"OpenAI test failed: {str(e)}")
        return False

def test_anthropic_key():
    """Test if the Anthropic API key is valid."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    assert api_key, "ANTHROPIC_API_KEY not found in environment variables"
    
    # Create the adapter
    adapter = AnthropicAdapter()
    
    # Simple test message
    messages = [{"role": "user", "content": "Hello, this is a test message."}]
    
    try:
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            model="claude-3-haiku-20240307",  # Using a smaller model for cost efficiency
            max_tokens=10  # Minimal response to save tokens
        )
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"Anthropic test passed: {response.get('content')}")
        return True
    except Exception as e:
        print(f"Anthropic test failed: {str(e)}")
        return False

def test_deepseek_key():
    """Test if the DeepSeek API key is valid."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    assert api_key, "DEEPSEEK_API_KEY not found in environment variables"
    
    # Create the adapter
    adapter = DeepSeekAdapter()
    
    # Simple test message
    messages = [{"role": "user", "content": "Hello, this is a test message."}]
    
    try:
        # Make a minimal request
        completion = adapter.chat_completion(
            messages=messages,
            max_tokens=10  # Minimal response to save tokens
        )
        # Extract response to verify we got something back
        response = adapter.extract_response(completion)
        assert response.get("content"), "No content in response"
        print(f"DeepSeek test passed: {response.get('content')}")
        return True
    except Exception as e:
        print(f"DeepSeek test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing API keys...")
    openai_result = test_openai_key()
    anthropic_result = test_anthropic_key()
    deepseek_result = test_deepseek_key()
    
    print("\nSummary:")
    print(f"OpenAI API key: {'✓ Valid' if openai_result else '✗ Invalid'}")
    print(f"Anthropic API key: {'✓ Valid' if anthropic_result else '✗ Invalid'}")
    print(f"DeepSeek API key: {'✓ Valid' if deepseek_result else '✗ Invalid'}")
