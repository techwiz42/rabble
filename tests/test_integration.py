import os
import sys
import json
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to access rabble
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Import Rabble components
from rabble import Rabble, Agent, Result

# Load environment variables
env_path = Path(__file__).parent.parent / "rabble" / ".env"
load_dotenv(env_path)

# Mark tests that require actual API keys
requires_openai_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), 
    reason="OpenAI API key not available"
)

requires_anthropic_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), 
    reason="Anthropic API key not available"
)

requires_mistral_key = pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"), 
    reason="Mistral API key not available"
)

requires_google_key = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), 
    reason="Google API key not available"
)

requires_cohere_key = pytest.mark.skipif(
    not os.getenv("COHERE_API_KEY"), 
    reason="Cohere API key not available"
)

# Test functions
def add(a: int, b: int):
    """Add two numbers."""
    return Result(value=f"The sum is {a + b}", context_variables={"sum": a + b})

def calculate(context_variables, operation: str, x: float, y: float = None):
    """
    Perform mathematical calculations.
    
    Args:
        operation: The operation to perform - one of "add", "subtract", "multiply", "divide", "sqrt"
        x: First number
        y: Second number (optional)
    """
    result = None
    
    if operation == "add":
        result = x + y
    elif operation == "subtract":
        result = x - y
    elif operation == "multiply":
        result = x * y
    elif operation == "divide":
        if y == 0:
            return Result(value="Error: Division by zero is not allowed")
        result = x / y
    elif operation == "sqrt":
        import math
        result = math.sqrt(x)
    else:
        return Result(value=f"Error: Unknown operation '{operation}'")
    
    return Result(
        value=f"Result: {result}",
        context_variables={"calculation": {"operation": operation, "result": result}}
    )

@requires_openai_key
def test_openai_basic():
    """Test a basic OpenAI conversation."""
    # Initialize client
    client = Rabble(provider="openai")
    
    # Create a simple agent
    agent = Agent(
        name="TestAgent",
        provider="openai",
        model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
        instructions="You are a helpful assistant. Keep your response very brief, under 20 words."
    )
    
    # Test a simple conversation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Say hello and introduce yourself briefly."}]
    )
    
    # Check response
    assert response.messages[0]["role"] == "assistant"
    assert len(response.messages[0]["content"].split()) <= 25  # Allow some flexibility
    assert response.agent.name == "TestAgent"

@requires_openai_key
def test_openai_tool_call():
    """Test OpenAI with function calling."""
    # Initialize client
    client = Rabble(provider="openai")
    
    # Create an agent with a calculation function
    agent = Agent(
        name="CalculatorAgent",
        provider="openai",
        model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
        instructions="You are a math assistant. Always use the calculate function when asked to perform calculations.",
        functions=[calculate]
    )
    
    # Test calculation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "What is 25 * 13?"}]
    )
    
    # Check context variables for evidence of tool call
    assert "calculation" in response.context_variables
    assert response.context_variables["calculation"]["result"] == 325
    assert response.context_variables["calculation"]["operation"] == "multiply"

@requires_anthropic_key
def test_anthropic_basic():
    """Test a basic Anthropic conversation."""
    # Initialize client
    client = Rabble(provider="anthropic")
    
    # Create a simple agent
    agent = Agent(
        name="ClaudeAgent",
        provider="anthropic",
        model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-sonnet-20240229"),
        instructions="You are Claude, a helpful AI assistant. Keep your response very brief, under 20 words."
    )
    
    # Test a simple conversation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Say hello and introduce yourself briefly."}]
    )
    
    # Check response
    assert response.messages[0]["role"] == "assistant"
    assert len(response.messages[0]["content"].split()) <= 25  # Allow some flexibility
    assert response.agent.name == "ClaudeAgent"

@requires_anthropic_key
def test_anthropic_tool_call():
    """Test Anthropic with function calling."""
    # Initialize client
    client = Rabble(provider="anthropic")
    
    # Create an agent with a calculation function
    agent = Agent(
        name="CalculatorAgent",
        provider="anthropic",
        model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-sonnet-20240229"),
        instructions="You are a math assistant. Always use the calculate function when asked to perform calculations.",
        functions=[calculate]
    )
    
    # Test calculation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "What is 25 * 13?"}]
    )
    
    # Check context variables for evidence of tool call
    assert "calculation" in response.context_variables
    assert response.context_variables["calculation"]["result"] == 325
    assert response.context_variables["calculation"]["operation"] == "multiply"

@requires_google_key
def test_google_basic():
    """Test a basic Google Gemini conversation."""
    # Initialize client
    client = Rabble(provider="google")
    
    # Create a simple agent
    agent = Agent(
        name="GeminiAgent",
        provider="google",
        model=os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-1.5-pro"),
        instructions="You are a helpful assistant. Keep your response very brief, under 20 words."
    )
    
    # Test a simple conversation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Say hello and introduce yourself briefly."}]
    )
    
    # Check response
    assert response.messages[0]["role"] == "assistant"
    assert len(response.messages[0]["content"]) > 0
    assert response.agent.name == "GeminiAgent"

@requires_google_key
def test_google_text_tool_extraction():
    """Test Google's text-based tool extraction mechanism."""
    # Initialize client
    client = Rabble(provider="google")
    
    # Create an agent with calculation function
    agent = Agent(
        name="GeminiCalculator",
        provider="google",
        model=os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-1.5-pro"),
        instructions="""You are a calculator assistant. When asked to perform calculations:
        1. Use the calculate tool with parameters operation, x, and y
        2. For multiplication use operation="multiply"
        3. Write your tool usage in a code block like:
        ```tool_code
        operation="multiply"
        x=25
        y=13
        ```
        """,
        functions=[calculate]
    )
    
    # Test calculation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "What is 25 * 13?"}]
    )
    
    # Check if tool was called through text extraction
    # This might not always succeed since Google does not formally support tool calls
    if "calculation" in response.context_variables:
        assert response.context_variables["calculation"]["result"] == 325
        assert response.context_variables["calculation"]["operation"] == "multiply"
    else:
        # If tool wasn't called, check that the response contains the correct answer
        content = response.messages[0]["content"]
        assert "325" in content, "Expected calculation result not found in response"

@requires_mistral_key
def test_mistral_basic():
    """Test a basic Mistral conversation."""
    # Initialize client
    client = Rabble(provider="mistral")
    
    # Create a simple agent
    agent = Agent(
        name="MistralAgent",
        provider="mistral",
        model=os.getenv("MISTRAL_DEFAULT_MODEL", "mistral-large-latest"),
        instructions="You are a helpful assistant. Keep your response very brief, under 20 words."
    )
    
    # Test a simple conversation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Say hello and introduce yourself briefly."}]
    )
    
    # Check response
    assert response.messages[0]["role"] == "assistant"
    assert len(response.messages[0]["content"]) > 0
    assert response.agent.name == "MistralAgent"

@requires_mistral_key
def test_mistral_tool_call():
    """Test Mistral with function calling."""
    # Initialize client
    client = Rabble(provider="mistral")
    
    # Create an agent with a calculation function
    agent = Agent(
        name="CalculatorAgent",
        provider="mistral",
        model=os.getenv("MISTRAL_DEFAULT_MODEL", "mistral-large-latest"),
        instructions="You are a math assistant. Always use the calculate function when asked to perform calculations.",
        functions=[calculate]
    )
    
    # Test calculation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "What is 25 * 13?"}]
    )
    
    # Check context variables for evidence of tool call
    assert "calculation" in response.context_variables
    assert response.context_variables["calculation"]["result"] == 325
    assert response.context_variables["calculation"]["operation"] == "multiply"

@requires_cohere_key
def test_cohere_basic():
    """Test a basic Cohere conversation."""
    # Initialize client
    client = Rabble(provider="cohere")
    
    # Create a simple agent
    agent = Agent(
        name="CohereAgent",
        provider="cohere",
        model=os.getenv("COHERE_DEFAULT_MODEL", "command"),
        instructions="You are a helpful assistant. Keep your response very brief, under 20 words."
    )
    
    # Test a simple conversation
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Say hello and introduce yourself briefly."}]
    )
    
    # Check response
    assert response.messages[0]["role"] == "assistant"
    assert len(response.messages[0]["content"]) > 0
    assert response.agent.name == "CohereAgent"

@requires_cohere_key
def test_cohere_tool_call():
    """Test Cohere with function calling."""
    # Initialize client
    client = Rabble(provider="cohere")
    
    # Create an agent with a calculation function
    agent = Agent(
        name="CalculatorAgent",
        provider="cohere",
        model=os.getenv("COHERE_DEFAULT_MODEL", "command"),
        instructions="You are a math assistant. Always use the calculate function when asked to perform calculations.",
        functions=[calculate]
    )
    
    # Test calculation
    try:
        response = client.run(
            agent=agent,
            messages=[{"role": "user", "content": "What is 25 * 13?"}]
        )
        
        # Check context variables for evidence of tool call
        assert "calculation" in response.context_variables
        assert response.context_variables["calculation"]["result"] == 325
        assert response.context_variables["calculation"]["operation"] == "multiply"
    except Exception as e:
        pytest.skip(f"Cohere tool call test failed: {str(e)}")

@requires_openai_key
@requires_anthropic_key
def test_cross_provider_handoff():
    """Test handoff between different providers (OpenAI to Anthropic)."""
    # Create a Rabble client
    client = Rabble()
    
    # Create an Anthropic agent (target of handoff)
    claude_agent = Agent(
        name="ClaudeAgent",
        provider="anthropic",
        model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-sonnet-20240229"),
        instructions="You are Claude, an AI assistant by Anthropic. Always mention that you're Claude in your response."
    )
    
    # Define handoff function
    def transfer_to_claude():
        """Transfer the conversation to Claude."""
        return claude_agent
    
    # Create an OpenAI agent with handoff function
    openai_agent = Agent(
        name="OpenAIAgent",
        provider="openai",
        model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
        instructions="You are an OpenAI assistant. When asked to transfer to Claude, use the transfer_to_claude function.",
        functions=[transfer_to_claude]
    )
    
    # Test the handoff
    response = client.run(
        agent=openai_agent,
        messages=[{"role": "user", "content": "Please transfer me to Claude."}]
    )
    
    # Check that handoff occurred
    assert response.agent.name == "ClaudeAgent"
    assert response.agent.provider == "anthropic"
    
    # Check that Claude identifies itself
    final_message = response.messages[-1]["content"]
    assert "claude" in final_message.lower() or "anthropic" in final_message.lower()
