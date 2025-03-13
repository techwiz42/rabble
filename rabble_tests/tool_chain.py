# examples/tool_chain_test.py
"""
A test that attempts to have each provider use a calculator tool in sequence.
If a provider successfully completes a tool call, its output becomes 
the prompt for the next provider.

Providers tested:
- OpenAI
- Anthropic
- Google
- Mistral
- Cohere
- DeepSeek
"""

import os
import sys
import time
import math
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

# Define a simple calculator function
def calculate(context_variables, operation: str, x: float, y: float) -> Result:
    """
    Perform a mathematical calculation on two numbers.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide, power, sqrt)
        x: The first number
        y: The second number (not used for sqrt operation)
        
    Returns:
        The result of the calculation
    """
    result = None
    operation = operation.lower()
    
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
    elif operation == "power":
        result = x ** y
        description = f"{x} ^ {y} = {result}"
    elif operation == "sqrt":
        result = math.sqrt(x)
        description = f"sqrt({x}) = {result}"
    else:
        return Result(value=f"Error: Unknown operation '{operation}'")
    
    # Return a nicely formatted result
    return Result(
        value=f"Calculation result: {description}",
        context_variables={"last_calculation": {"operation": operation, "x": x, "y": y, "result": result}}
    )

def run_tool_chain_test():
    """Test each provider's ability to use a tool function."""
    # Initialize the Rabble client
    client = Rabble()
    
    print("Multi-Provider Tool Call Chain Test")
    print("----------------------------------")
    print("Each provider is asked to use the calculator tool.")
    print("If successful, its response becomes the prompt for the next provider.")
    
    # Create an agent for each provider with identical tool
    agents = []
    
    # Define all providers with identical configuration except for provider name and model
    providers = [
        ("OpenAI", "openai", os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")),
        ("Anthropic", "anthropic", os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-7-sonnet-20250219")),
        ("Google", "google", os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-1.5-flash")),
        ("Mistral", "mistral", os.getenv("MISTRAL_DEFAULT_MODEL", "mistral-small-latest")),
        ("Cohere", "cohere", os.getenv("COHERE_DEFAULT_MODEL", "command-light"))
        #("DeepSeek", "deepseek", os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"))
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
                instructions="""You are a helpful assistant with access to a calculator tool.
                
When asked to perform a calculation, always use the calculate tool to get accurate results.
- For addition, use operation="add"
- For subtraction, use operation="subtract"
- For multiplication, use operation="multiply"
- For division, use operation="divide"
- For exponentiation, use operation="power"
- For square root, use operation="sqrt" (only x parameter is used)

After using the tool, briefly explain the calculation you performed.""",
                functions=[calculate]
            )
            agents.append(agent)
    
    # Print models being used
    print("\nProvider order:")
    for agent in agents:
        print(f"- {agent.name}: {agent.model}")
    
    # Define initial calculation prompts for each agent
    calculation_prompts = [
        "What is 17398 + 280551?",
        "What is 12320556 - 458871?",
        "What is 1253155 * 80675?",
        "What is 14402031 / 12717?",
        "What is 3 to the power of 4.5?",
        "What is the square root of 169714?",
        "What is 75 + 25?"
    ]
    
    # Track all responses
    responses = []
    
    # Run the chain, testing each agent with a tool call
    for i, agent in enumerate(agents):
        # Get the appropriate calculation prompt
        prompt = calculation_prompts[i % len(calculation_prompts)]
        
        # Add a small delay between API calls to avoid rate limits
        time.sleep(1)
        
        print(f"\n{agent.name} responding to: \"{prompt}\"")
        
        try:
            # Use the Rabble framework to call the agent with a timeout parameter
            # This will pass the timeout to all adapters that support it
            response = client.run(
                agent=agent,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                timeout=30  # 30 second timeout for each API call
            )
            
            # Check if a tool was used
            tool_messages = [msg for msg in response.messages if msg.get("role") == "tool"]
            tool_used = False
            
            if tool_messages:
                for tool_msg in tool_messages:
                    tool_name = tool_msg.get("tool_name", "Unknown Tool")
                    tool_result = tool_msg.get("content", "No result")
                    print(f"{agent.name} [Used {tool_name}] {tool_result}")
                    tool_used = True
            
            # Get the assistant's final response
            assistant_message = next((msg for msg in response.messages if msg.get("role") == "assistant"), None)
            if assistant_message and assistant_message.get("content"):
                content = assistant_message.get("content")
                print(f"{agent.name} responded: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")
                
                responses.append({
                    "agent": agent.name,
                    "prompt": prompt,
                    "response": content,
                    "tool_used": tool_used,
                    "calculation_result": response.context_variables.get("last_calculation", {})
                })
            else:
                print(f"{agent.name} didn't provide a proper response.")
                
        except TimeoutError as e:
            print(f"Timeout error with {agent.name}: {str(e)}")
            # Record the timeout error
            responses.append({
                "agent": agent.name,
                "prompt": prompt,
                "error": f"Timeout: {str(e)}",
                "tool_used": False
            })
        except Exception as e:
            print(f"Error with {agent.name}: {str(e)}")
            # Record the error
            responses.append({
                "agent": agent.name,
                "prompt": prompt,
                "error": str(e),
                "tool_used": False
            })
    
    # Print the summary of results
    print("\nTool Call Results Summary:")
    print("========================")
    
    for data in responses:
        agent = data["agent"]
        prompt = data["prompt"]
        
        print(f"\n{agent} - Prompt: \"{prompt}\"")
        
        if "error" in data:
            print(f"  ❌ Error: {data['error']}")
        elif data["tool_used"]:
            calc = data.get("calculation_result", {})
            if calc:
                result = calc.get("result", "unknown")
                operation = calc.get("operation", "unknown")
                x = calc.get("x", "unknown")
                y = calc.get("y", "unknown")
                
                print(f"  ✅ Tool used successfully")
                print(f"  → Operation: {operation}")
                print(f"  → Inputs: x={x}, y={y}")
                print(f"  → Result: {result}")
            else:
                print(f"  ✅ Tool used but no calculation result stored")
        else:
            print(f"  ❌ No tool used")
        
        # Print a snippet of the response if available
        if "response" in data:
            response = data["response"]
            print(f"  Response: \"{response[:100]}{'...' if len(response) > 100 else ''}\"")

if __name__ == "__main__":
    run_tool_chain_test()
