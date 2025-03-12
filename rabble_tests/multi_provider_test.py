# examples/complete_multi_provider_test.py
"""
A comprehensive test demonstrating the use of all seven language model providers in a single workflow.

This example creates a chain of agents from all supported providers, with each agent:
1. Performing a specific task using a tool
2. Storing the result in context variables
3. Passing control to the next agent in the chain

Providers tested:
- OpenAI (GPT)
- Anthropic (Claude)
- Google (Gemini)
- Mistral AI
- Cohere
- DeepSeek
- Together AI

Requirements:
- API keys for each provider in your .env file
- pip install rabble openai anthropic mistralai cohere google-generativeai python-dotenv requests
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add rabble to path if needed
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from rabble import Rabble, Agent
from rabble.types import Result

# Load environment variables
env_path = Path(__file__).parent.parent / "rabble" / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"Environment file not found at {env_path}")

load_dotenv(env_path)

# Verify all required environment variables are present
required_env_vars = [
    # API Keys
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", 
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    
    # Default Models
    "OPENAI_DEFAULT_MODEL",
    "ANTHROPIC_DEFAULT_MODEL",
    "GOOGLE_DEFAULT_MODEL",
    "MISTRAL_DEFAULT_MODEL",
    "COHERE_DEFAULT_MODEL",
    "TOGETHER_DEFAULT_MODEL",
    "DEEPSEEK_DEFAULT_MODEL"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Tool functions for the agents to use

def analyze_text(text: str, aspect: str) -> str:
    """
    Analyze the given text for a specific aspect or characteristic.
    
    Args:
        text: The text to analyze
        aspect: The aspect to analyze (sentiment, complexity, theme, etc.)
    
    Returns:
        Analysis of the text for the requested aspect
    """
    return f"Analysis of '{text[:20]}...' for {aspect}: This is a simulated analysis."

def calculate_statistics(context_variables, numbers: str) -> Result:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: A comma-separated list of numbers
        
    Returns:
        Basic statistics (min, max, mean)
    """
    try:
        nums = [float(n.strip()) for n in numbers.split(",")]
        stats = {
            "min": min(nums),
            "max": max(nums),
            "mean": sum(nums) / len(nums),
            "count": len(nums)
        }
        result = f"Statistics: min={stats['min']}, max={stats['max']}, mean={stats['mean']}, count={stats['count']}"
        
        # Store in context variables for next agent
        return Result(
            value=result,
            context_variables={"statistics": stats}
        )
    except Exception as e:
        return f"Error calculating statistics: {str(e)}"

def generate_response_template(context_variables, template_type: str) -> Result:
    """
    Generate a response template of the specified type.
    
    Args:
        template_type: The type of template to generate (email, report, etc.)
        
    Returns:
        A template with placeholders
    """
    templates = {
        "email": "Subject: {{subject}}\n\nDear {{name}},\n\n{{body}}\n\nBest regards,\n{{sender}}",
        "report": "# {{title}}\n\n## Executive Summary\n{{summary}}\n\n## Details\n{{details}}\n\n## Conclusion\n{{conclusion}}",
        "announcement": "ANNOUNCEMENT: {{title}}\n\n{{details}}\n\nDate: {{date}}"
    }
    
    template = templates.get(template_type, "Template type not recognized")
    
    # Store the template in context variables for the next agent
    return Result(
        value=f"Generated template for '{template_type}':\n\n{template}",
        context_variables={"template": template, "template_type": template_type}
    )

def categorize_items(context_variables, items: str, categories: str) -> Result:
    """
    Categorize a list of items into the specified categories.
    
    Args:
        items: Comma-separated list of items
        categories: Comma-separated list of categories
        
    Returns:
        Items organized into categories
    """
    item_list = [item.strip() for item in items.split(",")]
    category_list = [cat.strip() for cat in categories.split(",")]
    
    # Simple demo categorization - alternates categories
    categorized = {}
    for i, category in enumerate(category_list):
        categorized[category] = [item for j, item in enumerate(item_list) if j % len(category_list) == i]
    
    result = "Categorized items:\n"
    for category, items in categorized.items():
        result += f"\n{category}: {', '.join(items)}"
    
    # Store in context variables for the next agent
    return Result(
        value=result,
        context_variables={"categorized_items": categorized}
    )

def summarize_text(context_variables, text: str, max_length: int = 100) -> Result:
    """
    Summarize the provided text to the specified maximum length.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary
        
    Returns:
        A summarized version of the text
    """
    # Simple demo summarization - just truncates
    summary = text[:max_length] + "..." if len(text) > max_length else text
    
    # Store in context variables for the next agent
    return Result(
        value=f"Summary: {summary}",
        context_variables={"summary": summary}
    )

def extract_data(context_variables, text: str, data_type: str) -> Result:
    """
    Extract structured data of the specified type from text.
    
    Args:
        text: The text to extract data from
        data_type: Type of data to extract (dates, names, numbers, etc.)
        
    Returns:
        Extracted data
    """
    # Simple demo extraction
    extraction = f"Extracted {data_type} from text: Simulated extraction result"
    
    # Store in context variables for the next agent
    return Result(
        value=extraction,
        context_variables={"extracted_data": extraction, "data_type": data_type}
    )

def generate_quiz(context_variables, topic: str, num_questions: int = 3) -> Result:
    """
    Generate a quiz on the specified topic.
    
    Args:
        topic: The topic for the quiz
        num_questions: Number of questions to generate
        
    Returns:
        A quiz with questions and answers
    """
    # Demo quiz generation
    quiz = {
        "topic": topic,
        "questions": [
            {"question": f"Sample question {i+1} about {topic}?", "answer": f"Sample answer {i+1}"} 
            for i in range(int(num_questions))
        ]
    }
    
    result = f"Quiz on {topic}:\n"
    for i, q in enumerate(quiz["questions"]):
        result += f"\n{i+1}. {q['question']}\n   Answer: {q['answer']}"
    
    # Store in context variables for the next agent
    return Result(
        value=result,
        context_variables={"quiz": quiz}
    )

# Agent transfer functions
def transfer_to_openai():
    """Transfer to OpenAI agent."""
    return openai_agent

def transfer_to_anthropic():
    """Transfer to Anthropic agent."""
    return anthropic_agent

def transfer_to_google():
    """Transfer to Google agent."""
    return google_agent

def transfer_to_mistral():
    """Transfer to Mistral agent."""
    return mistral_agent

def transfer_to_cohere():
    """Transfer to Cohere agent."""
    return cohere_agent

def transfer_to_together():
    """Transfer to Together agent."""
    return together_agent

def transfer_to_deepseek():
    """Transfer to DeepSeek agent."""
    return deepseek_agent

# Create agents for each provider
# Each agent gets model from environment variables (NO FALLBACKS)

openai_agent = Agent(
    name="OpenAI Assistant",
    provider="openai",
    model=os.getenv("OPENAI_DEFAULT_MODEL"),
    instructions="""You are an OpenAI assistant specialized in text analysis.
    
    Use the analyze_text tool to analyze the user's input.
    After completing your analysis, transfer to the Anthropic assistant.""",
    functions=[analyze_text, transfer_to_anthropic]
)

anthropic_agent = Agent(
    name="Anthropic Assistant",
    provider="anthropic",
    model=os.getenv("ANTHROPIC_DEFAULT_MODEL"),
    instructions="""You are an Anthropic Claude assistant specialized in statistical analysis.
    
    Use the calculate_statistics tool to analyze numerical data.
    After completing your analysis, transfer to the Google assistant.""",
    functions=[calculate_statistics, transfer_to_google]
)

google_agent = Agent(
    name="Google Assistant",
    provider="google",
    model=os.getenv("GOOGLE_DEFAULT_MODEL"),
    instructions="""You are a Google Gemini assistant specialized in template generation.
    
    Use the generate_response_template tool to create a template.
    After creating the template, transfer to the Mistral assistant.""",
    functions=[generate_response_template, transfer_to_mistral]
)

mistral_agent = Agent(
    name="Mistral Assistant",
    provider="mistral",
    model=os.getenv("MISTRAL_DEFAULT_MODEL"),
    instructions="""You are a Mistral AI assistant specialized in categorization.
    
    Use the categorize_items tool to organize items into categories.
    After completing the categorization, transfer to the Cohere assistant.""",
    functions=[categorize_items, transfer_to_cohere]
)

cohere_agent = Agent(
    name="Cohere Assistant",
    provider="cohere",
    model=os.getenv("COHERE_DEFAULT_MODEL"),
    instructions="""You are a Cohere assistant specialized in summarization.
    
    Use the summarize_text tool to create concise summaries.
    After completing your summary, transfer to the Together assistant.""",
    functions=[summarize_text, transfer_to_together]
)

together_agent = Agent(
    name="Together Assistant",
    provider="together",
    model=os.getenv("TOGETHER_DEFAULT_MODEL"),
    instructions="""You are a Together AI assistant specialized in data extraction.
    
    Use the extract_data tool to extract specific information from text.
    After completing the extraction, transfer to the DeepSeek assistant.""",
    functions=[extract_data, transfer_to_deepseek]
)

deepseek_agent = Agent(
    name="DeepSeek Assistant",
    provider="deepseek",
    model=os.getenv("DEEPSEEK_DEFAULT_MODEL"),
    instructions="""You are a DeepSeek assistant specialized in educational content.
    
    Use the generate_quiz tool to create a quiz on a given topic.
    After generating the quiz, transfer to the OpenAI assistant to complete the cycle.""",
    functions=[generate_quiz, transfer_to_openai]
)

def run_multi_provider_test():
    """Run the test of all providers with tool usage and agent handoffs."""
    # Initialize the Rabble client
    client = Rabble()
    
    print("Multi-Provider Chain Test with Tool Usage")
    print("----------------------------------------")
    print("This test demonstrates all seven providers in sequence, each:")
    print("1. Using a specialized tool")
    print("2. Storing results in context variables")
    print("3. Passing control to the next provider")
    print("\nProviders: OpenAI → Anthropic → Google → Mistral → Cohere → Together → DeepSeek → OpenAI")
    print("\nModels used:")
    for provider in ["OPENAI", "ANTHROPIC", "GOOGLE", "MISTRAL", "COHERE", "TOGETHER", "DEEPSEEK"]:
        print(f"- {provider}: {os.getenv(f'{provider}_DEFAULT_MODEL')}")
    print()
    
    # Start the chain with OpenAI analyzing text
    messages = [{
        "role": "user", 
        "content": "Let's test all providers in sequence. Can you analyze this text for its sentiment: 'I'm really enjoying this multi-provider test of different language models!'"
    }]
    
    print("User: Let's test all providers in sequence. Can you analyze this text for its sentiment: 'I'm really enjoying this multi-provider test of different language models!'")
    
    # Initial context variables
    context_variables = {}
    
    # Track the chain of responses
    all_responses = []
    current_agent = openai_agent
    
    # Run the full chain
    for i in range(10):  # Limit to 10 steps to prevent infinite loops
        response = client.run(
            agent=current_agent, 
            messages=messages,
            context_variables=context_variables
        )
        
        # Display response
        print(f"\n{response.agent.name}:", end=" ")
        
        # Check for tool usage and print tool input/output
        tool_messages = [msg for msg in response.messages if msg.get("role") == "tool"]
        if tool_messages:
            for tool_msg in tool_messages:
                tool_name = tool_msg.get("tool_name", "Unknown Tool")
                tool_result = tool_msg.get("content", "No result")
                print(f"[Used {tool_name}] {tool_result}")
        else:
            # Just print the assistant's message content if no tool was used
            print(response.messages[-1].get("content", ""))
        
        # Update context variables
        context_variables = response.context_variables
        
        # Update messages for next step
        messages = response.messages
        
        # Check if we've made a complete cycle back to OpenAI
        if i > 0 and response.agent.name == "OpenAI Assistant":
            print("\nCompleted full cycle through all providers!")
            break
            
        # Break the loop if agent didn't change (should not happen with proper transfers)
        if response.agent == current_agent:
            print("\nWarning: Agent did not transfer control. Breaking loop.")
            break
            
        # Update current agent for next iteration
        current_agent = response.agent
        
        # Add a prompt to use the next tool in the chain
        if response.agent == anthropic_agent:
            messages.append({
                "role": "user", 
                "content": "Now, calculate statistics for these numbers: 10, 25, 15, 35, 20"
            })
            print("\nUser: Now, calculate statistics for these numbers: 10, 25, 15, 35, 20")
        elif response.agent == google_agent:
            messages.append({
                "role": "user", 
                "content": "Please generate an email template"
            })
            print("\nUser: Please generate an email template")
        elif response.agent == mistral_agent:
            messages.append({
                "role": "user", 
                "content": "Categorize these items: apple, python, banana, java, orange, ruby into two categories: fruits, programming languages"
            })
            print("\nUser: Categorize these items: apple, python, banana, java, orange, ruby into two categories: fruits, programming languages")
        elif response.agent == cohere_agent:
            messages.append({
                "role": "user", 
                "content": "Summarize this text: 'Multi-provider orchestration allows developers to combine the strengths of different language models from various providers. This approach provides flexibility, eliminates vendor lock-in, and enables more specialized workflows.'"
            })
            print("\nUser: Summarize this text: 'Multi-provider orchestration allows developers to combine the strengths of different language models from various providers. This approach provides flexibility, eliminates vendor lock-in, and enables more specialized workflows.'")
        elif response.agent == together_agent:
            messages.append({
                "role": "user", 
                "content": "Extract dates from this text: 'The project started on January 15, 2024 and is scheduled to be completed by March 30, 2024.'"
            })
            print("\nUser: Extract dates from this text: 'The project started on January 15, 2024 and is scheduled to be completed by March 30, 2024.'")
        elif response.agent == deepseek_agent:
            messages.append({
                "role": "user", 
                "content": "Generate a short quiz about machine learning"
            })
            print("\nUser: Generate a short quiz about machine learning")
        
        # Store response
        all_responses.append(response)
    
    # Print final context variables
    print("\nFinal Context Variables:")
    for key, value in context_variables.items():
        print(f"- {key}: {value}")
    
    return all_responses

if __name__ == "__main__":
    run_multi_provider_test()
