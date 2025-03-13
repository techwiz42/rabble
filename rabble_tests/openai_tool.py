#!/usr/bin/env python
"""
OpenAI Adapter Test Harness

This script tests the OpenAI adapter with both creative writing
and mathematical calculations using tool calls.
"""

import os
import sys
import math
import json
import random
import time
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to access rabble
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Import from rabble
from rabble import Rabble, Agent
from rabble.types import Result
from rabble.adapters.openai_adapter import OpenAIAdapter

# Load environment variables
env_path = Path(__file__).parent.parent / "rabble" / ".env"
load_dotenv(env_path)

# Ensure we have the required environment variables
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

print(f"Using OpenAI model: {model}")

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

def test_openai_creative_writing():
    """Test OpenAI's creative writing capabilities without tool use."""
    print("\n=== Testing OpenAI Creative Writing ===\n")
    
    # Create Rabble client with OpenAI adapter
    adapter = OpenAIAdapter(default_model=model)
    client = Rabble(client=adapter, provider="openai", model=model)
    
    # Create agent
    agent = Agent(
        name="OpenAI Poet",
        provider="openai",
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
            print(f"OpenAI's poem ({elapsed_time:.2f}s):")
            print(f"\n{content}\n")
        else:
            print("No response received")
            
        return True
    except Exception as e:
        print(f"Error in creative writing test: {str(e)}")
        return False

def test_openai_calculation():
    """Test OpenAI's calculation abilities using tool calls."""
    print("\n=== Testing OpenAI Tool Use with Calculations ===\n")
    
    # Create Rabble client with OpenAI adapter
    adapter = OpenAIAdapter(default_model=model)
    client = Rabble(client=adapter, provider="openai", model=model)
    
    # Create agent with calculator function
    agent = Agent(
        name="OpenAI Calculator",
        provider="openai",
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
        print("Sending request to OpenAI...")
        response = client.run(
            agent=agent,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            debug=debug
        )
        elapsed_time = time.time() - start_time
        
        # Check for tool usage
        tool_messages = [msg for msg in response.messages if msg.get("role") == "tool"]
        tool_used = False
        
        if tool_messages:
            print(f"Tool usage ({elapsed_time:.2f}s):")
            for tool_msg in tool_messages:
                tool_name = tool_msg.get("tool_name", "Unknown Tool")
                tool_result = tool_msg.get("content", "No result")
                print(f"  {tool_name}: {tool_result}")
                tool_used = True
        
        # Get and print the final response
        assistant_message = response.messages[-1] if response.messages and response.messages[-1].get("role") == "assistant" else None
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
            
        return tool_used
    except Exception as e:
        print(f"Error in calculation test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_openai_streaming():
    """Test OpenAI's streaming capabilities."""
    print("\n=== Testing OpenAI Streaming ===\n")
    
    # Create Rabble client with OpenAI adapter
    adapter = OpenAIAdapter(default_model=model)
    client = Rabble(client=adapter, provider="openai", model=model)
    
    # Create agent with calculator function
    agent = Agent(
        name="OpenAI Stream",
        provider="openai",
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

if __name__ == "__main__":
    print("OpenAI Adapter Test Harness")
    print("===========================")
    
    # Set debug environment variable
    os.environ["OPENAI_DEBUG"] = "1"
    
    # Run tests
    creative_result = test_openai_creative_writing()
    calculation_result = test_openai_calculation()
    streaming_result = test_openai_streaming()
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"  Creative Writing Test: {'✓ Passed' if creative_result else '✗ Failed'}")
    print(f"  Calculation Test:      {'✓ Passed' if calculation_result else '✗ Failed'}")
    print(f"  Streaming Test:        {'✓ Passed' if streaming_result else '✗ Failed'}")
