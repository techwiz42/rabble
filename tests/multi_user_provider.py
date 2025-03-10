# tests/test_multi_provider.py
import os
import pytest
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add rabble directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rabble import Rabble, Agent
from rabble.types import Result

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "rabble" / ".env")

# Skip tests if API keys are not available
def check_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    # We can run tests with just these two providers
    return bool(openai_key and anthropic_key)

pytestmark = pytest.mark.skipif(not check_keys(), reason="API keys not available")

# Test functions for the agents to use
def summarize_text(text: str):
    """Summarize the given text."""
    return f"I've analyzed: {text[:20]}... (Test function, not real summarization)"

def analyze_with_claude():
    """Switch to Claude for detailed analysis."""
    return claude_agent

def return_to_gpt():
    """Switch back to GPT for general assistance."""
    return openai_agent

def store_analysis(context_variables, analysis: str):
    """Store analysis in context variables."""
    return Result(
        value="Analysis stored successfully.",
        context_variables={"analysis": analysis}
    )

# Define the agents
openai_agent = Agent(
    name="GPT Assistant",
    provider="openai",
    model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo"),
    instructions="You are a helpful assistant. Keep responses very short for testing purposes.",
    functions=[summarize_text, analyze_with_claude, store_analysis]
)

claude_agent = Agent(
    name="Claude Assistant",
    provider="anthropic",
    model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-haiku-20240307"),
    instructions="You are Claude. Keep responses very short for testing purposes.",
    functions=[return_to_gpt, store_analysis]
)

@pytest.fixture
def rabble_client():
    return Rabble()

def test_basic_openai(rabble_client):
    """Test basic interaction with OpenAI."""
    messages = [{"role": "user", "content": "Say hello briefly"}]
    response = rabble_client.run(agent=openai_agent, messages=messages)
    
    assert response.agent.name == "GPT Assistant"
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"], "No content in response"
    print(f"OpenAI response: {response.messages[-1]['content']}")

def test_basic_anthropic(rabble_client):
    """Test basic interaction with Anthropic."""
    messages = [{"role": "user", "content": "Say hello briefly"}]
    response = rabble_client.run(agent=claude_agent, messages=messages)
    
    assert response.agent.name == "Claude Assistant"
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"], "No content in response"
    print(f"Claude response: {response.messages[-1]['content']}")

def test_handoff_between_providers(rabble_client):
    """Test handoff from OpenAI to Anthropic and back."""
    messages = [{"role": "user", "content": "Say hello, then analyze with Claude"}]
    
    # First response from OpenAI
    response = rabble_client.run(agent=openai_agent, messages=messages)
    print(f"1. {response.agent.name}: {response.messages[-1]['content']}")
    
    # Should have called analyze_with_claude function
    messages = response.messages
    assert response.agent.name == "Claude Assistant", "Did not switch to Claude"
    
    # Continue with Claude
    messages.append({"role": "user", "content": "Now return to GPT"})
    response = rabble_client.run(agent=response.agent, messages=messages)
    print(f"2. {response.agent.name}: {response.messages[-1]['content']}")
    
    # Should have called return_to_gpt function
    assert response.agent.name == "GPT Assistant", "Did not switch back to GPT"
    
    # Verify we can continue the conversation
    messages = response.messages
    messages.append({"role": "user", "content": "Summarize our conversation"})
    response = rabble_client.run(agent=response.agent, messages=messages)
    print(f"3. {response.agent.name}: {response.messages[-1]['content']}")
    
    assert response.messages[-1]["content"], "No content in final response"
    
    return True

if __name__ == "__main__":
    if not check_keys():
        print("Skipping tests: API keys not available")
        sys.exit(0)
        
    # Create the client
    client = Rabble()
    
    # Run the tests manually
    print("Testing basic OpenAI interaction...")
    test_basic_openai(client)
    
    print("\nTesting basic Anthropic interaction...")
    test_basic_anthropic(client)
    
    print("\nTesting handoff between providers...")
    test_handoff_between_providers(client)
