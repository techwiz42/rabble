# examples/prompt_chain.py
"""
A simple test that passes a prompt from one provider to the next using the Rabble framework,
with each provider's response becoming the prompt for the next provider.

This script demonstrates the consistent API across different providers.

Providers tested:
- OpenAI
- Anthropic
- Google
- Mistral
- Cohere
- Together
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

def run_prompt_chain():
    """Run a chain of prompts across different providers."""
    # Initialize the Rabble client
    client = Rabble()
    
    print("Simple Prompt Chain Across Providers")
    print("-----------------------------------")
    print("Each provider's response becomes the prompt for the next provider.")
    print("This demonstrates the consistent API across different providers.")
    
    # Create an agent for each provider
    agents = []
    
    # Define all providers with identical configuration except for provider name and model
    providers = [
        ("OpenAI", "openai", os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")),
        ("Anthropic", "anthropic", os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-7-sonnet-20250219")),
        ("Google", "google", os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-1.5-flash")),
        ("Mistral", "mistral", os.getenv("MISTRAL_DEFAULT_MODEL", "mistral-small-latest")),
        ("Cohere", "cohere", os.getenv("COHERE_DEFAULT_MODEL", "command-light")),
        ("DeepSeek", "deepseek", os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"))
    ]
    
    # Create agents using the same configuration pattern for all providers
    for name, provider, model in providers:
        # Only create agent if API key exists
        key_var = f"{provider.upper()}_API_KEY"
        if os.getenv(key_var):
            agent = Agent(
                name=f"{name} Agent",
                provider=provider,
                model=model,
                instructions="You are a helpful assistant. Respond to the user's prompt in a creative and engaging way."
            )
            agents.append(agent)
    
    # Print models being used
    print("\nProvider order:")
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
        
        print(f"\n{agent.name} responding to: \"{current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}\"")
        
        try:
            # Use exactly the same API call for all providers
            response = client.run(
                agent=agent,
                messages=[{"role": "user", "content": current_prompt}],
                max_tokens=500
            )
            
            # Get the assistant's response
            assistant_message = next((msg for msg in response.messages if msg.get("role") == "assistant"), None)
            if assistant_message and assistant_message.get("content"):
                content = assistant_message.get("content")
                print(f"{agent.name} responded: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")
                
                responses.append({
                    "agent": agent.name,
                    "prompt": current_prompt,
                    "response": content
                })
                
                # Update prompt for next agent
                current_prompt = content
            else:
                print(f"{agent.name} didn't provide a proper response.")
                fallback_prompt = current_prompt + "\n\nNote: The previous agent couldn't respond. Please provide your own response."
                print(f"Using fallback prompt for next agent.")
                current_prompt = fallback_prompt
                
        except Exception as e:
            print(f"Error with {agent.name}: {str(e)}")
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
