#!/usr/bin/env python
"""
Google Adapter Test Harness

This script tests the Google adapter with creative writing,
mathematical calculations, streaming, and external tool use.
"""

import os
import sys
import math
import json
import random
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to access rabble
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Import from rabble
from rabble import Rabble, Agent
from rabble.types import Result
from rabble.adapters.google_adapter import GoogleAdapter

# Load environment variables
env_path = Path(__file__).parent.parent / "rabble" / ".env"
load_dotenv(env_path)

# Ensure we have the required environment variables
api_key = os.getenv("GOOGLE_API_KEY")
model = os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-1.5-pro")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

print(f"Using Google model: {model}")

# Define a calculator function that can do various operations
def calculate(context_variables, operation: str, x: float, y: float = None) -> Result:
    """
    Perform mathematical calculations.
    
    Args:
        operation: The operation to perform - one of "add", "subtract", "multiply", "divide", "sqrt"
        x: First number
        y: Second number (not required for sqrt operation)
        
    Returns:
        The result of the calculation
    """
    result = None
    description = ""
    
    try:
        if operation == "add":
            result = x + y
            description = f"{x} + {y} = {result}"
        elif operation == "subtract":
            result = x - y
            description = f"{x} - {y} = {result}"
        elif operation == "multiply":
            result = x * y
            description = f"{x} * {y} = {result}"
        elif operation == "divide":
            if y == 0:
                return Result(value="Error: Division by zero is not allowed")
            result = x / y
            description = f"{x} / {y} = {result}"
        elif operation == "sqrt":
            if x < 0:
                return Result(value="Error: Cannot take square root of negative number")
            result = math.sqrt(x)
            description = f"sqrt({x}) = {result}"
        else:
            return Result(value=f"Error: Unknown operation '{operation}'")
    except Exception as e:
        return Result(value=f"Error in calculation: {str(e)}")
    
    return Result(
        value=f"Calculation result: {description}",
        context_variables={
            "last_calculation": {
                "operation": operation,
                "x": x,
                "y": y,
                "result": result
            }
        }
    )

def get_weather(context_variables, location: str) -> Result:
    """
    Get the current weather for a specific location.
    
    Args:
        location: City name or location (e.g., "New York", "Tokyo", "London")
        
    Returns:
        Weather information for the specified location
    """
    # This function returns fake data, but Google has no way of knowing that
    # It can't generate this data internally since it would need API access
    
    # Create a unique, unpredictable response based on the location string
    # Use a hash of the location to generate "random" but deterministic values
    
    # Create a hash of the location
    location_hash = hashlib.md5(location.lower().encode()).hexdigest()
    
    # Convert parts of the hash to weather values (this is deterministic but unpredictable)
    temp_value = int(location_hash[:2], 16) % 40  # Temperature between 0-39°C
    humidity = int(location_hash[2:4], 16) % 100  # Humidity between 0-99%
    
    # Map some values to weather conditions
    conditions = [
        "Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorms", 
        "Snowy", "Foggy", "Windy", "Clear", "Overcast"
    ]
    condition_index = int(location_hash[4:6], 16) % len(conditions)
    condition = conditions[condition_index]
    
    # Generate a weather ID that's impossible to predict
    weather_id = location_hash[8:16]
    
    # Create the response
    weather_data = {
        "location": location,
        "temperature": f"{temp_value}°C",
        "humidity": f"{humidity}%",
        "condition": condition,
        "weather_id": weather_id  # This would be impossible for Google to predict
    }
    
    return Result(
        value=f"Weather for {location}: {condition}, {temp_value}°C with {humidity}% humidity (Weather ID: {weather_id})",
        context_variables={"weather_data": weather_data}
    )

def test_google_creative_writing():
    """Test Google's creative writing capabilities without tool use."""
    print("\n=== Testing Google Creative Writing ===\n")
    
    # Create Rabble client with Google adapter
    adapter = GoogleAdapter(default_model=model)
    client = Rabble(client=adapter, provider="google", model=model)
    
    # Create agent
    agent = Agent(
        name="Google Poet",
        provider="google",
        model=model,
        instructions="You are a thoughtful AI poet. When asked to write poetry, create insightful and meaningful verse."
    )
    
    # Prompt for a poem
    prompt = "Write a small poem (4-8 lines) about the relationship between humans and artificial intelligence."
    
    print(f"Prompt: \"{prompt}\"\n")
    
    try:
        # Make request
        start_time = time.time()
        response = client.run(
            agent=agent,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        elapsed_time = time.time() - start_time
        
        # Get and print the response
        assistant_message = next((msg for msg in response.messages if msg.get("role") == "assistant"), None)
        if assistant_message and assistant_message.get("content"):
            content = assistant_message.get("content")
            print(f"Google's poem ({elapsed_time:.2f}s):")
            print(f"\n{content}\n")
        else:
            print("No response received")
            
        return True
    except Exception as e:
        print(f"Error in creative writing test: {str(e)}")
        return False

def test_google_calculation():
    """Test Google's calculation abilities using tool calls."""
    print("\n=== Testing Google Tool Use with Calculations ===\n")
    
    # Create Rabble client with Google adapter
    adapter = GoogleAdapter(default_model=model)
    client = Rabble(client=adapter, provider="google", model=model)
    
    # Create agent with calculator function
    agent = Agent(
        name="Google Calculator",
        provider="google",
        model=model,
        instructions="""You are a calculator assistant with access to a calculator tool.
        
When asked to perform calculations, always use the calculate tool with precise inputs:
- For addition, use operation="add" with x and y parameters
- For subtraction, use operation="subtract" with x and y parameters
- For multiplication, use operation="multiply" with x and y parameters
- For division, use operation="divide" with x and y parameters
- For square root, use operation="sqrt" with only the x parameter

Always verify your inputs and explain your calculation approach before using the tool.""",
        functions=[calculate]
    )
    
    # Generate two random 6-digit numbers
    num1 = random.randint(100000, 999999)
    num2 = random.randint(100000, 999999)
    
    # Create the calculation prompt
    prompt = f"""Perform these calculations step by step:
1. Multiply these two numbers: {num1} and {num2}
2. Take the square root of their product

Please show your work and the final result."""
    
    print(f"Numbers: {num1} and {num2}")
    print(f"Expected product: {num1 * num2}")
    print(f"Expected square root: {math.sqrt(num1 * num2)}")
    print(f"\nPrompt: \"{prompt}\"\n")
    
    # Enable debugging for verbose output
    debug = True
    
    try:
        # Make request
        start_time = time.time()
        print("Sending request to Google...")
        response = client.run(
            agent=agent,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            debug=debug
        )
        elapsed_time = time.time() - start_time
        
        # Check for tool usage
        tool_messages = [msg for msg in response.messages if msg.get("role") == "tool"]
        actual_tool_used = len(tool_messages) > 0
        
        if tool_messages:
            print(f"Tool usage ({elapsed_time:.2f}s):")
            for tool_msg in tool_messages:
                tool_name = tool_msg.get("tool_name", "Unknown Tool")
                tool_result = tool_msg.get("content", "No result")
                print(f"  {tool_name}: {tool_result}")
        
        # Check for described tool usage
        assistant_message = response.messages[-1] if response.messages and response.messages[-1].get("role") == "assistant" else None
        described_tool_used = False
        
        if assistant_message and assistant_message.get("tool_calls"):
            described_tool_used = True
            print(f"Extracted tool calls from text:")
            for tool_call in assistant_message["tool_calls"]:
                print(f"  - {tool_call['function']['name']}({tool_call['function']['arguments']})")
        
        # Get and print the final response
        if assistant_message and assistant_message.get("content"):
            content = assistant_message.get("content")
            print(f"\nFinal response ({elapsed_time:.2f}s):")
            print(f"{content}")
        else:
            print("No final response received")
        
        # Check calculation results in context variables
        calc_results = response.context_variables.get("last_calculation", {})
        if calc_results:
            print("\nCalculation results from context variables:")
            print(f"  Operation: {calc_results.get('operation')}")
            print(f"  Input: {calc_results.get('x')} {'' if calc_results.get('operation') == 'sqrt' else 'and ' + str(calc_results.get('y'))}")
            print(f"  Result: {calc_results.get('result')}")
        
        # Consider the test successful if either actual tools were used OR
        # the adapter successfully extracted tool calls from text
        return actual_tool_used or described_tool_used
        
    except Exception as e:
        print(f"Error in calculation test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_google_streaming():
    """Test Google's streaming capabilities."""
    print("\n=== Testing Google Streaming ===\n")
    
    # Create Rabble client with Google adapter
    adapter = GoogleAdapter(default_model=model)
    client = Rabble(client=adapter, provider="google", model=model)
    
    # Create agent with calculator function
    agent = Agent(
        name="Google Stream",
        provider="google",
        model=model,
        instructions="You are a helpful assistant that provides concise responses.",
        functions=[calculate]
    )
    
    # Simple prompt to test streaming
    prompt = "Explain the relationship between mathematics and AI in 3-4 sentences."
    
    print(f"Prompt: \"{prompt}\"\n")
    print("Streaming response:")
    
    try:
        # Make streaming request
        stream = client.run(
            agent=agent,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=300
        )
        
        # Process stream
        content = ""
        for chunk in stream:
            if "content" in chunk and chunk["content"] is not None:
                print(chunk["content"], end="", flush=True)
                content += chunk["content"]
            
            if "delim" in chunk and chunk["delim"] == "end" and content:
                print()
                content = ""
            
            if "response" in chunk:
                print("\nStream complete.")
                return True
        
        return True
    except Exception as e:
        print(f"\nError in streaming test: {str(e)}")
        return False

def test_google_external_tool():
    """Test Google's ability to call an external tool that requires data it can't generate internally."""
    print("\n=== Testing Google External Tool Use ===\n")
    
    # Create Rabble client with Google adapter
    adapter = GoogleAdapter(default_model=model)
    client = Rabble(client=adapter, provider="google", model=model)
    
    # Create agent with weather function
    agent = Agent(
        name="Google Weather Assistant",
        provider="google",
        model=model,
        instructions="""You are a weather assistant with access to a weather API tool.
        
When asked about weather in any location, you must use the get_weather tool to get accurate data.
You do not have weather information for any location without using this tool.
Each location has a unique Weather ID that can only be obtained by using the tool.

IMPORTANT: Never make up or guess weather data. Always use the get_weather tool.""",
        functions=[get_weather]
    )
    
    # Query for weather data
    prompt = "What's the current weather in Singapore? I need the exact temperature, humidity, condition, and especially the Weather ID."
    
    print(f"Prompt: \"{prompt}\"\n")
    
    # Enable debugging for verbose output
    debug = True
    
    try:
        # Make request
        start_time = time.time()
        print("Sending request to Google...")
        response = client.run(
            agent=agent,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            debug=debug
        )
        elapsed_time = time.time() - start_time
        
        # Check for actual tool usage
        tool_messages = [msg for msg in response.messages if msg.get("role") == "tool"]
        actual_tool_used = len(tool_messages) > 0
        
        # Check for described tool usage (tool_code blocks or text descriptions)
        assistant_message = next((msg for msg in response.messages if msg.get("role") == "assistant"), None)
        described_tool_used = False
        if assistant_message and assistant_message.get("tool_calls"):
            described_tool_used = True
        
        # Check for weather ID in the response
        weather_id_found = False
        content = ""
        if assistant_message and assistant_message.get("content"):
            content = assistant_message.get("content")
            weather_id_found = "Weather ID:" in content or "weather_id" in content.lower()
        
        # Print results
        print(f"\nResults ({elapsed_time:.2f}s):")
        print(f"  Actual Tool Called: {'✓ Yes' if actual_tool_used else '✗ No'}")
        print(f"  Tool Usage Described: {'✓ Yes' if described_tool_used else '✗ No'}")
        print(f"  Weather ID Mentioned: {'✓ Yes' if weather_id_found else '✗ No'}")
        
        # Print the response content
        print("\nResponse Content:")
        print(content)
        
        # Check weather data in context variables
        weather_data = response.context_variables.get("weather_data", {})
        if weather_data:
            print("\nWeather data from context variables:")
            for key, value in weather_data.items():
                print(f"  {key}: {value}")
        
        # A true pass requires the actual tool to be called
        return actual_tool_used
        
    except Exception as e:
        print(f"Error in external tool test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Google Adapter Test Harness")
    print("==========================")
    
    # Set debug environment variable
    os.environ["GOOGLE_DEBUG"] = "1"
    
    # Run tests
    creative_result = test_google_creative_writing()
    calculation_result = test_google_calculation()
    streaming_result = test_google_streaming()
    external_tool_result = test_google_external_tool()
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"  Creative Writing Test: {'✓ Passed' if creative_result else '✗ Failed'}")
    print(f"  Calculation Test:      {'✓ Passed' if calculation_result else '✗ Failed'}")
    print(f"  Streaming Test:        {'✓ Passed' if streaming_result else '✗ Failed'}")
    print(f"  External Tool Test:    {'✓ Passed' if external_tool_result else '✗ Failed'}")
