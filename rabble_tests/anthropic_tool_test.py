#!/usr/bin/env python
"""
Anthropic Tool Test

This script tests the Anthropic API with tools based on the current Claude tool format.
"""

import os
import json
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

# Test with the correct tool format based on Anthropic's documentation
try:
    print("\n=== Testing Anthropic Tool Format ===\n")
    from anthropic import Anthropic
    
    client = Anthropic(api_key=api_key)
    
    # Define the tool with the correct format for current Claude API
    weather_tool = {
        "name": "get_weather",
        "description": "Get the weather for a location. Returns current temperature, conditions, and other weather information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature"
                }
            },
            "required": ["location"]
        }
    }
    
    print("Sending request with tool...")
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        tools=[weather_tool]
    )
    
    print("\nResponse content:")
    tool_use_found = False
    
    for block in response.content:
        if hasattr(block, 'type'):
            if block.type == 'text':
                print(f"Text: {block.text}")
            elif block.type == 'tool_use':
                tool_use_found = True
                print(f"\nTool use detected:")
                print(f"  Tool ID: {block.id}")
                print(f"  Tool name: {block.name}")
                print(f"  Tool input: {json.dumps(block.input, indent=2)}")
    
    # If a tool was used, now let's respond to it
    if tool_use_found:
        print("\n=== Testing Tool Response ===\n")
        
        # Find the tool use block
        tool_use_block = next((block for block in response.content if hasattr(block, 'type') and block.type == 'tool_use'), None)
        
        if tool_use_block:
            tool_id = tool_use_block.id
            
            # Create a fake weather result
            weather_result = {
                "temperature": 72,
                "condition": "sunny",
                "precipitation": 0,
                "location": tool_use_block.input.get("location", "San Francisco"),
                "unit": tool_use_block.input.get("unit", "fahrenheit")
            }
            
            # Create message chain with tool result in Anthropic's format
            messages = [
                {"role": "user", "content": "What's the weather like in San Francisco?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", 
                     "id": tool_id, 
                     "name": tool_use_block.name, 
                     "input": tool_use_block.input}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", 
                     "tool_call_id": tool_id, 
                     "content": json.dumps(weather_result)}
                ]}
            ]
            
            print("Sending tool result back to Claude...")
            continuation = client.messages.create(
                model=model,
                max_tokens=300,
                messages=messages,
                tools=[weather_tool]  # Include the same tools again
            )
            
            print("\nFinal response content:")
            for block in continuation.content:
                if hasattr(block, 'type') and block.type == 'text':
                    print(f"Text: {block.text}")

except Exception as e:
    print(f"Error in Anthropic tool test: {e}")
    import traceback
    traceback.print_exc()

# Now let's test with our updated Rabble adapter
try:
    print("\n=== Testing with Updated Rabble Adapter ===\n")
    
    # Import the necessary components
    from rabble import Rabble, Agent
    from rabble.types import Result
    
    # Define a weather function
    def get_weather(context_variables, location: str, unit: str = "fahrenheit") -> Result:
        """
        Get the weather for a location.
        
        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: The unit of temperature (celsius or fahrenheit)
        """
        # Simulate getting weather data
        weather_data = {
            "temperature": 72,
            "condition": "sunny",
            "precipitation": 0,
            "location": location,
            "unit": unit
        }
        
        return Result(
            value=f"Weather in {location}: {weather_data['temperature']}°{unit[0].upper()}, {weather_data['condition']}",
            context_variables={"weather_data": weather_data}
        )
    
    # Create an agent with the function
    agent = Agent(
        name="Claude Weather Assistant",
        provider="anthropic",
        model=model,
        instructions="You are a helpful weather assistant. Use the get_weather function to look up weather information.",
        functions=[get_weather]
    )
    
    # Create Rabble client
    client = Rabble()
    
    # Run a query that should trigger the function
    print("Sending request to Rabble framework...")
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "What's the weather like in Chicago?"}],
        max_tokens=300
    )
    
    print("\nRabble response:")
    print(f"Agent: {response.agent.name}")
    
    # Check for tool usage in the messages
    tool_messages = [msg for msg in response.messages if msg.get("role") == "tool"]
    if tool_messages:
        print("\nTool messages:")
        for tool_msg in tool_messages:
            print(f"  Tool: {tool_msg.get('tool_name', 'Unknown')}")
            print(f"  Result: {tool_msg.get('content', 'No result')}")
    
    print(f"\nFinal response: {response.messages[-1]['content']}")
    
except Exception as e:
    print(f"Error in Rabble framework test: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Tests Complete ===")
