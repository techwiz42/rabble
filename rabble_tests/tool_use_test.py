# examples/simple_chained_calculations.py
"""
A simplified test of multiple language model providers performing the same math calculation
without agent transfers. Each provider uses the output values from the previous provider.

Providers tested:
- OpenAI (GPT)
- Google (Gemini) 
- Mistral AI
- Cohere
- Together AI
- DeepSeek
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from dotenv import load_dotenv

# Add rabble to path if needed
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from rabble import Rabble, Agent
from rabble.types import Result

# Load environment variables
env_path = Path(__file__).parent.parent / "rabble" / ".env"
load_dotenv(env_path)

# Chained math function for all agents to use
def calculate_math(context_variables, num1: float, num2: float) -> Result:
    """
    Calculate mathematical operations on two numbers.
    
    Args:
        num1: The first number
        num2: The second number
        
    Returns:
        Results including the square root of the sum of squares and the average
    """
    # Calculate the square root of the sum of squares (Euclidean norm)
    sum_of_squares = num1**2 + num2**2
    euclidean_norm = math.sqrt(sum_of_squares)
    
    # Calculate the average
    average = (num1 + num2) / 2
    
    # Create result data
    results = {
        "num1": num1,
        "num2": num2,
        "sum_of_squares": sum_of_squares,
        "euclidean_norm": euclidean_norm,
        "average": average
    }
    
    # Format a readable response
    response = f"Math results:\n- Input numbers: {num1} and {num2}\n- Sum of squares: {sum_of_squares}\n- Euclidean norm: {euclidean_norm:.4f}\n- Average: {average}"
    
    return Result(
        value=response,
        context_variables={"last_calculation": results}
    )

def perform_calculation(client, agent, num1, num2):
    """Perform calculation with specified agent and input values."""
    print(f"\n{agent.name} calculating with values {num1:.4f} and {num2:.4f}...")
    
    messages = [{
        "role": "user", 
        "content": f"Use the calculate_math function with num1={num1} and num2={num2}."
    }]
    
    try:
        response = client.run(
            agent=agent,
            messages=messages,
            max_tokens=300
        )
        
        # Check for tool usage
        tool_messages = [msg for msg in response.messages if msg.get("role") == "tool"]
        if tool_messages:
            for tool_msg in tool_messages:
                tool_name = tool_msg.get("tool_name", "Unknown Tool")
                tool_result = tool_msg.get("content", "No result")
                print(f"{agent.name} [Used {tool_name}] {tool_result}")
                
                # Return the calculation results if available
                if "last_calculation" in response.context_variables:
                    return response.context_variables["last_calculation"]
        else:
            # The model didn't use the tool
            print(f"{agent.name} didn't use the calculate_math function. Response: {response.messages[-1].get('content', '')}")
            
        return None
    except Exception as e:
        print(f"Error with {agent.name}: {str(e)}")
        return None

def run_simple_chain():
    """Run a simple chain of calculations across different providers."""
    # Initialize the Rabble client
    client = Rabble()
    
    print("Simple Chained Calculations Across Providers")
    print("-------------------------------------------")
    print("Each provider performs the same calculation, taking as input")
    print("the output values from the previous provider.")
    print("\nProviders: OpenAI → Google → Mistral → Cohere → Together → DeepSeek")
    
    # Create an agent for each provider (without transfer functions)
    agents = []
    
    # OpenAI Agent
    if os.getenv("OPENAI_API_KEY"):
        openai_agent = Agent(
            name="OpenAI Agent",
            provider="openai",
            model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
            instructions="You are an OpenAI assistant. Use the calculate_math function with the exact values provided.",
            functions=[calculate_math]
        )
        agents.append(openai_agent)
    
    # Google Agent
    if os.getenv("GOOGLE_API_KEY"):
        google_agent = Agent(
            name="Google Agent",
            provider="google",
            model=os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-1.5-flash"),
            instructions="You are a Google Gemini assistant. Use the calculate_math function with the exact values provided.",
            functions=[calculate_math]
        )
        agents.append(google_agent)
    
    # Mistral Agent
    if os.getenv("MISTRAL_API_KEY"):
        # Extra debugging for Mistral
        print(f"Mistral API Key (first 5 chars): {os.getenv('MISTRAL_API_KEY')[:5]}***")
        mistral_agent = Agent(
            name="Mistral Agent",
            provider="mistral",
            model=os.getenv("MISTRAL_DEFAULT_MODEL", "mistral-small-latest"),
            instructions="You are a Mistral AI assistant. Use the calculate_math function with the exact values provided.",
            functions=[calculate_math]
        )
        agents.append(mistral_agent)
    
    # Cohere Agent
    if os.getenv("COHERE_API_KEY"):
        cohere_agent = Agent(
            name="Cohere Agent",
            provider="cohere",
            model=os.getenv("COHERE_DEFAULT_MODEL", "command-light"),
            instructions="You are a Cohere assistant. Use the calculate_math function with the exact values provided.",
            functions=[calculate_math]
        )
        agents.append(cohere_agent)
    
    # DeepSeek Agent
    if os.getenv("DEEPSEEK_API_KEY"):
        deepseek_agent = Agent(
            name="DeepSeek Agent",
            provider="deepseek",
            model=os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
            instructions="You are a DeepSeek assistant. Use the calculate_math function with the exact values provided.",
            functions=[calculate_math]
        )
        agents.append(deepseek_agent)
    
    # Print models being used
    print("\nModels:")
    for agent in agents:
        print(f"- {agent.name}: {agent.model}")
    
    # Starting values
    num1, num2 = 1, 2
    print(f"\nStarting values: {num1} and {num2}")
    
    # Track all calculations
    calculations = []
    
    # Perform the calculation chain
    current_num1, current_num2 = num1, num2
    
    for agent in agents:
        # Add a small delay between API calls to avoid rate limits
        time.sleep(1)
        
        calculation = perform_calculation(client, agent, current_num1, current_num2)
        
        if calculation:
            calculations.append({
                "agent": agent.name,
                "calculation": calculation
            })
            
            # Update values for next agent
            current_num1 = calculation["euclidean_norm"]
            current_num2 = calculation["average"]
        else:
            print(f"Skipping to next agent due to error with {agent.name}")
    
    # Print the calculation chain
    print("\nCalculation Chain:")
    print("=================")
    
    # Starting values
    print(f"Starting values: {num1} and {num2}")
    
    # Print each step in the calculation chain
    for i, calc_data in enumerate(calculations):
        agent = calc_data["agent"]
        calc = calc_data["calculation"]
        print(f"\nStep {i+1} ({agent}):")
        print(f"  Input:  {calc['num1']:.4f} and {calc['num2']:.4f}")
        print(f"  Euclidean norm: {calc['euclidean_norm']:.4f}")
        print(f"  Average: {calc['average']:.4f}")
        
        # Show what will be passed to the next agent
        if i < len(calculations) - 1:
            print(f"  → Passing {calc['euclidean_norm']:.4f} and {calc['average']:.4f} to next agent")

if __name__ == "__main__":
    run_simple_chain()
