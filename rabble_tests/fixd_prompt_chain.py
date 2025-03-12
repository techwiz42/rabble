# examples/simple_prompt_chain.py
"""
A simple test that passes a prompt from one provider to the next,
with each provider's response becoming the prompt for the next provider.

Providers tested:
- OpenAI (GPT)
- Anthropic (Claude)
- Google (Gemini)
- Mistral AI
- Cohere
- Together AI
- DeepSeek
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add rabble to path if needed
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from rabble import Rabble, Agent

# Load environment variables
env_path = Path(__file__).parent.parent / "rabble" / ".env"
load_dotenv(env_path)

def get_provider_response(client, agent, prompt):
    """Get a response from an agent with the given prompt."""
    print(f"\n{agent.name} responding to: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
    
    messages = [{
        "role": "user", 
        "content": prompt
    }]
    
    try:
        response = client.run(
            agent=agent,
            messages=messages,
            max_tokens=500
        )
        
        # Get the assistant's response
        assistant_message = next((msg for msg in response.messages if msg.get("role") == "assistant"), None)
        if assistant_message and assistant_message.get("content"):
            content = assistant_message.get("content")
            print(f"{agent.name} responded: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")
            return content
        else:
            print(f"{agent.name} didn't provide a proper response.")
            return None
    except Exception as e:
        print(f"Error with {agent.name}: {str(e)}")
        return None

def run_prompt_chain():
    """Run a chain of prompts across different providers."""
    # Initialize the Rabble client
    client = Rabble()
    
    print("Simple Prompt Chain Across All Providers")
    print("---------------------------------------")
    print("Each provider's response becomes the prompt for the next provider.")
    print("\nProvider order: OpenAI → Anthropic → Google → Mistral → Cohere → Together → DeepSeek")
    
    # Create an agent for each provider
    agents = []
    
    # OpenAI Agent
    if os.getenv("OPENAI_API_KEY"):
        openai_agent = Agent(
            name="OpenAI Agent",
            provider="openai",
            model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
            instructions="You are a helpful assistant. Respond to the user's prompt in a creative and engaging way."
        )
        agents.append(openai_agent)
    
    # Anthropic Agent
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_agent = Agent(
            name="Anthropic Agent",
            provider="anthropic",
            model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-7-sonnet-20250219"),
            instructions="You are a helpful assistant. Respond to the user's prompt in a creative and engaging way."
        )
        agents.append(anthropic_agent)
    
    # Google Agent
    if os.getenv("GOOGLE_API_KEY"):
        google_agent = Agent(
            name="Google Agent",
            provider="google",
            model=os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-1.5-flash"),
            instructions="You are a helpful assistant. Respond to the user's prompt in a creative and engaging way."
        )
        agents.append(google_agent)
    
    # Mistral Agent
    if os.getenv("MISTRAL_API_KEY"):
        mistral_agent = Agent(
            name="Mistral Agent",
            provider="mistral",
            model=os.getenv("MISTRAL_DEFAULT_MODEL", "mistral-small-latest"),
            instructions="You are a helpful assistant. Respond to the user's prompt in a creative and engaging way."
        )
        agents.append(mistral_agent)
    
    # Cohere Agent
    if os.getenv("COHERE_API_KEY"):
        cohere_agent = Agent(
            name="Cohere Agent",
            provider="cohere",
            model=os.getenv("COHERE_DEFAULT_MODEL", "command-light"),
            instructions="You are a helpful assistant. Respond to the user's prompt in a creative and engaging way."
        )
        agents.append(cohere_agent)
    
    # DeepSeek Agent
    if os.getenv("DEEPSEEK_API_KEY"):
        deepseek_agent = Agent(
            name="DeepSeek Agent",
            provider="deepseek",
            model=os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
            instructions="You are a helpful assistant. Respond to the user's prompt in a creative and engaging way."
        )
        agents.append(deepseek_agent)
    
    # Print models being used
    print("\nModels:")
    for agent in agents:
        print(f"- {agent.name}: {agent.model}")
    
    # Starting prompt
    initial_prompt = "Write a short poem about artificial intelligence."
    print(f"\nStarting prompt: \"{initial_prompt}\"")
    
    # Track all responses
    responses = []
    
    # Run the prompt chain
    current_prompt = initial_prompt
    
    for agent in agents:
        # Add a small delay between API calls to avoid rate limits
        time.sleep(1)
        
        response = get_provider_response(client, agent, current_prompt)
        
        if response:
            responses.append({
                "agent": agent.name,
                "prompt": current_prompt,
                "response": response
            })
            
            # Update prompt for next agent
            current_prompt = response
        else:
            # If an agent fails, use the previous prompt + notice for the next agent
            fallback_prompt = current_prompt + "\n\nNote: The previous agent couldn't respond. Please provide your own response."
            print(f"Using fallback prompt for next agent due to error.")
            current_prompt = fallback_prompt
    
    # Print the full chain of responses
    print("\nComplete Prompt Chain:")
    print("=====================")
    
    print(f"Initial prompt: \"{initial_prompt}\"\n")
    
    for i, data in enumerate(responses):
        agent = data["agent"]
        response = data["response"]
        
        print(f"Step {i+1}: {agent}")
        print(f"{response}\n")
        
        if i < len(responses) - 1:
            print("↓ Passed to next agent ↓\n")

if __name__ == "__main__":
    run_prompt_chain()
