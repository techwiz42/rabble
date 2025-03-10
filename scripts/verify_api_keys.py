#!/usr/bin/env python
# scripts/verify_keys.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import time
import argparse

# Add rabble directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from rabble.adapters import ModelAdapterFactory

def verify_key(provider, model=None):
    """Test if the API key for a specific provider is valid."""
    env_var_name = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(env_var_name)
    
    if not api_key:
        return False, f"{env_var_name} not found in environment variables"
    
    # Get default model from env or use a reasonable default
    if not model:
        model = os.getenv(f"{provider.upper()}_DEFAULT_MODEL")
    
    try:
        # Create the adapter
        adapter = ModelAdapterFactory.create_adapter(
            provider=provider,
            model=model
        )
        
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
        if not response.get("content"):
            return False, "No content in response"
        
        return True, response.get("content")
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Verify API keys for LLM providers")
    parser.add_argument("--provider", choices=["openai", "anthropic", "deepseek", "all"], 
                      default="all", help="Provider to test (default: all)")
    parser.add_argument("--env-file", default=os.path.join(parent_dir, "rabble", ".env"),
                      help="Path to .env file (default: rabble/.env)")
    
    args = parser.parse_args()
    
    # Load environment variables
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        print(f"Loaded environment from {args.env_file}")
    else:
        print(f"Warning: .env file not found at {args.env_file}")
    
    providers = ["openai", "anthropic", "deepseek"] if args.provider == "all" else [args.provider]
    
    results = {}
    
    print("Testing API keys...\n")
    
    for provider in providers:
        print(f"Testing {provider}... ", end="", flush=True)
        success, message = verify_key(provider)
        results[provider] = (success, message)
        
        if success:
            print("✓ Success")
            print(f"  Response: {message[:50]}{'...' if len(message) > 50 else ''}\n")
        else:
            print("✗ Failed")
            print(f"  Error: {message}\n")
        
        # Small delay to avoid rate limits
        time.sleep(1)
    
    # Print summary
    print("\nSummary:")
    for provider, (success, _) in results.items():
        status = "✓ Valid" if success else "✗ Invalid"
        print(f"{provider.capitalize()} API key: {status}")

if __name__ == "__main__":
    main()
