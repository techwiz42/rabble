#!/usr/bin/env python
"""
Minimal Anthropic Tool Format Debug

Tests a series of tool formats with Anthropic to find what works.
"""

import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent / "rabble" / ".env"
load_dotenv(env_path)

api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_DEFAULT_MODEL")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
if not model:
    raise ValueError("ANTHROPIC_DEFAULT_MODEL not found in environment variables")

print(f"Using model: {model}")

from anthropic import Anthropic
client = Anthropic(api_key=api_key)

# Test a series of tool formats, starting with extremely minimal and adding features gradually
formats = [
    # Format 1: Absolutely minimal format
    [{
        "type": "custom",
        "custom": {
            "name": "get_weather"
        }
    }],
    
    # Format 2: Adding input_schema
    [{
        "type": "custom",
        "custom": {
            "name": "get_weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string"
                    }
                },
                "required": ["location"]
            }
        }
    }],
    
    # Format 3: Adding description
    [{
        "type": "custom",
        "custom": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string"
                    }
                },
                "required": ["location"]
            }
        }
    }],
    
    # Format 4: More property details
    [{
        "type": "custom",
        "custom": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }]
]

# Test each format
for i, tool_format in enumerate(formats):
    print(f"\n=== Testing Format {i+1} ===\n")
    print(json.dumps(tool_format, indent=2))
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            tools=tool_format
        )
        
        print("\nSUCCESS!")
        # Check if tool was used
        tool_used = False
        for block in response.content:
            if hasattr(block, 'type') and block.type == 'tool_use':
                tool_used = True
                print(f"Tool used: {block.tool_use.name}")
                print(f"Input: {json.dumps(block.tool_use.input, indent=2)}")
            elif hasattr(block, 'type') and block.type == 'text':
                print(f"Text response: {block.text[:50]}...")
        
        if not tool_used:
            print("Tool was not used in the response.")
        
    except Exception as e:
        print(f"ERROR: {e}")

print("\nDebug complete!")
