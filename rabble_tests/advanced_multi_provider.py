# examples/advanced_multi_provider.py
"""
An example demonstrating the use of multiple language model providers in a single workflow.

This example creates a chain of agents from different providers that can hand off tasks to each other
based on their strengths:
- OpenAI (GPT-4o) for general-purpose questions
- Anthropic (Claude) for detailed reasoning
- Mistral for coding tasks
- Cohere for summarization

Requirements:
- API keys for each provider in your .env file
- pip install rabble openai anthropic mistralai cohere python-dotenv
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add rabble to path if needed
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from rabble import Rabble, Agent
from rabble.types import Result

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "rabble" / ".env")

# Initialize common functionality for all agents
def summarize_input(text: str):
    """Summarize the provided text in a few sentences."""
    return f"I'll summarize the text: {text[:30]}... [This is a placeholder - actual function would do real summarization]"

def store_result(context_variables, result: str, metadata: str = ""):
    """Store a result in the context variables for later use."""
    return Result(
        value="Result stored successfully.",
        context_variables={"result": result, "metadata": metadata}
    )

# Agent transfer functions
def transfer_to_claude():
    """Transfer to Claude for detailed reasoning and explanation."""
    return claude_agent

def transfer_to_gpt():
    """Transfer back to GPT for general assistance."""
    return gpt_agent

def transfer_to_mistral():
    """Transfer to Mistral for code generation."""
    return mistral_agent

def transfer_to_cohere():
    """Transfer to Cohere for summarization."""
    return cohere_agent

# Create agents for different providers
gpt_agent = Agent(
    name="GPT Assistant",
    provider="openai",
    model="gpt-4o",
    instructions="""You are a helpful assistant that specializes in routing requests to the most appropriate agent.
    
    For detailed reasoning tasks, transfer to Claude.
    For programming tasks, transfer to Mistral.
    For summarization tasks, transfer to Cohere.
    Otherwise, handle the request yourself.
    
    Keep responses concise and helpful.""",
    functions=[
        summarize_input, 
        store_result, 
        transfer_to_claude, 
        transfer_to_mistral, 
        transfer_to_cohere
    ]
)

claude_agent = Agent(
    name="Claude Reasoning Assistant",
    provider="anthropic",
    model="claude-3-5-sonnet-20240620",
    instructions="""You are Claude, an AI assistant specialized in detailed reasoning and explanation.
    
    When asked to explain complex topics, break them down step by step.
    Focus on clear, nuanced thinking.
    Once your explanation is complete, return to the GPT assistant.""",
    functions=[summarize_input, store_result, transfer_to_gpt]
)

mistral_agent = Agent(
    name="Mistral Coding Assistant",
    provider="mistral",
    model="mistral-large-latest",
    instructions="""You are a Mistral AI coding assistant.
    
    Provide clean, efficient code solutions.
    Include comments to explain your approach.
    Once your coding task is complete, return to the GPT assistant.""",
    functions=[summarize_input, store_result, transfer_to_gpt]
)

cohere_agent = Agent(
    name="Cohere Summarization Assistant",
    provider="cohere",
    model="command",
    instructions="""You are a Cohere AI summarization assistant.
    
    Create concise, accurate summaries of provided content.
    Maintain the key points while reducing length.
    Once summarization is complete, return to the GPT assistant.""",
    functions=[summarize_input, store_result, transfer_to_gpt]
)

def run_example():
    # Initialize the Rabble client
    client = Rabble()
    
    print("Multi-Provider Agent Chain Example")
    print("----------------------------------")
    print("This example demonstrates using multiple providers in a single workflow.")
    print("Available Agents:")
    print(" - GPT Assistant (OpenAI): General routing and assistance")
    print(" - Claude Assistant (Anthropic): Detailed reasoning and explanation")
    print(" - Mistral Assistant: Code generation")
    print(" - Cohere Assistant: Summarization")
    print()
    
    # Start with a general request to GPT
    messages = [{"role": "user", "content": "I'd like to understand quantum computing basics."}]
    
    # GPT should route this to Claude for detailed explanation
    print("User: I'd like to understand quantum computing basics.")
    response = client.run(agent=gpt_agent, messages=messages)
    print(f"{response.agent.name}: {response.messages[-1]['content']}")
    
    # Continue the conversation, asking for code
    messages = response.messages
    messages.append({"role": "user", "content": "Can you write a simple quantum random number generator in Python using Qiskit?"})
    
    # This should route to Mistral for code
    print("\nUser: Can you write a simple quantum random number generator in Python using Qiskit?")
    response = client.run(agent=response.agent, messages=messages)
    print(f"{response.agent.name}: {response.messages[-1]['content']}")
    
    # Ask for a summary of the conversation
    messages = response.messages
    messages.append({"role": "user", "content": "Can you summarize our conversation so far?"})
    
    # This should route to Cohere for summarization
    print("\nUser: Can you summarize our conversation so far?")
    response = client.run(agent=response.agent, messages=messages)
    print(f"{response.agent.name}: {response.messages[-1]['content']}")
    
    # Final message back to GPT
    messages = response.messages
    messages.append({"role": "user", "content": "Thank you for all the information!"})
    
    # This should stay with GPT
    print("\nUser: Thank you for all the information!")
    response = client.run(agent=response.agent, messages=messages)
    print(f"{response.agent.name}: {response.messages[-1]['content']}")
    
    return response

if __name__ == "__main__":
    run_example()
