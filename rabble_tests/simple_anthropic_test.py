#!/usr/bin/env python
"""
Simple Anthropic API Test

This script tests basic functionality with the Anthropic API without using tools.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add rabble directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Load environment variables
env_path = Path(__file__).parent.parent / "rabble" / ".env"
load_dotenv(env_path)

# Check for API key
api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_DEFAULT_MODEL")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
if not model:
    raise ValueError("ANTHROPIC_DEFAULT_MODEL not found in environment variables")

print(f"Using model: {model}")

# Test 1: Direct API call using the Anthropic library
try:
    print("\n=== Test 1: Direct Anthropic API Call ===\n")
    from anthropic import Anthropic
    
    client = Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[
            {"role": "user", "content": "Hello, please give me a short greeting."}
        ]
    )
    
    print("Response content:")
    for block in response.content:
        if hasattr(block, 'type') and block.type == 'text':
            print(f"Text block: {block.text}")
        else:
            print(f"Non-text block: {block.type if hasattr(block, 'type') else 'unknown'}")
    
    print("\nResponse metadata:")
    print(f"ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Stop reason: {response.stop_reason}")
    
except Exception as e:
    print(f"Error in direct API call: {e}")

# Test 2: Using the Rabble framework without tools
try:
    print("\n=== Test 2: Rabble Framework Call (No Tools) ===\n")
    
    from rabble import Rabble, Agent
    
    # Create a simple agent
    agent = Agent(
        name="Claude Test",
        provider="anthropic",
        model=model,
        instructions="You are a helpful assistant. Keep responses short and to the point."
    )
    
    # Create Rabble client
    client = Rabble()
    
    # Run a simple query
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Hello, please give me a short greeting."}],
        max_tokens=300  # This will now be passed through to the adapter
    )
    
    print("Rabble response:")
    print(f"Agent: {response.agent.name}")
    print(f"Content: {response.messages[-1]['content']}")
    
except Exception as e:
    print(f"Error in Rabble framework call: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Tests Complete ===")
