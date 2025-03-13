import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import json
from pathlib import Path

# Add parent directory to path to access rabble
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Import Rabble components
from rabble.adapters import (
    ModelAdapterFactory,
    OpenAIAdapter,
    AnthropicAdapter,
    MistralAdapter,
    GoogleAdapter,
    CohereAdapter
)
from rabble.util import function_to_json

# Mock function for testing tool format
def mock_function(x: int, y: int = 10):
    """A mock function for testing."""
    return x + y

class TestAdapters(unittest.TestCase):
    """Test all model adapters."""
    
    def test_adapter_factory(self):
        """Test the ModelAdapterFactory."""
        # Test creating OpenAI adapter
        with patch('openai.OpenAI'):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key", "OPENAI_DEFAULT_MODEL": "gpt-4o"}):
                adapter = ModelAdapterFactory.create_adapter(
                    provider="openai",
                    model="gpt-4o"
                )
                self.assertIsInstance(adapter, OpenAIAdapter)
        
        # Test creating Anthropic adapter
        with patch('anthropic.Anthropic'):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key", "ANTHROPIC_DEFAULT_MODEL": "claude-3-sonnet"}):
                adapter = ModelAdapterFactory.create_adapter(
                    provider="anthropic"
                )
                self.assertIsInstance(adapter, AnthropicAdapter)
        
        # Test unsupported provider
        with self.assertRaises(ValueError):
            ModelAdapterFactory.create_adapter(provider="unsupported")
    
    @patch('openai.OpenAI')
    def test_openai_adapter(self, mock_openai):
        """Test OpenAI adapter functionality."""
        # Configure mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create adapter
        adapter = OpenAIAdapter(client=mock_client, default_model="gpt-4o")
        
        # Test chat completion
        adapter.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[function_to_json(mock_function)],
            stream=False
        )
        
        # Verify client was called
        mock_client.chat.completions.create.assert_called_once()
        
        # Test format_tools
        tools = [function_to_json(mock_function)]
        formatted_tools = adapter.format_tools(tools)
        # OpenAI adapter should return tools as-is
        self.assertEqual(tools, formatted_tools)
    
    @patch('anthropic.Anthropic')
    def test_anthropic_adapter(self, mock_anthropic):
        """Test Anthropic adapter functionality."""
        # Configure mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Mock environment variables
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            # Create adapter
            adapter = AnthropicAdapter(client=mock_client, default_model="claude-3-sonnet")
            
            # Test chat completion
            adapter.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                tools=[function_to_json(mock_function)],
                stream=False
            )
            
            # Verify client was called
            mock_client.messages.create.assert_called_once()
            
            # Test format_tools - Anthropic has a different format
            tools = [function_to_json(mock_function)]
            formatted_tools = adapter.format_tools(tools)
            # Anthropic tools should use the input_schema format
            self.assertEqual(formatted_tools[0]["name"], "mock_function")
            self.assertIn("input_schema", formatted_tools[0])
        
    def test_google_adapter_simple(self):
        """Simple test for Google adapter text extraction functionality."""
        # Load environment variables from the correct location
        from dotenv import load_dotenv
        from pathlib import Path
        
        # Load .env file from the rabble directory
        env_path = Path(__file__).parent.parent / "rabble" / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment from {env_path}")
        else:
            print(f"Warning: .env file not found at {env_path}")
        
        # Get the actual API key and model from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-1.5-pro")
        
        # Skip test if no API key is available
        if not api_key:
            self.skipTest("Google API key not available")
            
        # Print information about what we're using (with masked key for security)
        masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:] if api_key else None
        print(f"Using Google API key: {masked_key}")
        print(f"Using Google model: {model}")
        
        # Skip this test if we're missing dependencies
        try:
            import google.generativeai as genai
        except ImportError:
            self.skipTest("google.generativeai package not installed")
            
        # Create an adapter with real API key but don't call any API methods
        adapter = GoogleAdapter(api_key=api_key, default_model=model)
        
        # Create a dictionary-based mock completion instead of using MagicMock
        # This avoids JSON serialization issues
        mock_completion = type('MockCompletion', (), {
            'text': """I'll calculate that for you.
        
First, I need to multiply 5 and 10:
```tool_code
operation="multiply"
x=5
y=10
```

The result is 50.
"""
        })
        
        adapter._current_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate things",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string"},
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "required": ["operation", "x"]
                    }
                }
            }
        ]
        
        # Only test the text-based extraction which doesn't rely on complex mocks
        response = adapter.extract_response(mock_completion)
        
        # Should extract a tool call with name "calculate"
        self.assertIsNotNone(response["tool_calls"])
        self.assertEqual(response["tool_calls"][0]["function"]["name"], "calculate")
        
        # Parse arguments to verify extraction
        args = json.loads(response["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["operation"], "multiply")
        self.assertEqual(args["x"], 5)
        self.assertEqual(args["y"], 10)

    @patch('cohere.Client')
    def test_cohere_adapter(self, mock_cohere_class):
        """Test Cohere adapter functionality."""
        # Configure mock
        mock_client = MagicMock()
        mock_cohere_class.return_value = mock_client
        
        # Mock environment variables
        with patch.dict(os.environ, {"COHERE_API_KEY": "fake-key", "COHERE_DEFAULT_MODEL": "command"}):
            # Create adapter
            adapter = CohereAdapter(client=mock_client, default_model="command")
            
            # Test chat completion
            adapter.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                tools=[function_to_json(mock_function)],
                stream=False
            )
            
            # Verify client was called
            mock_client.chat.assert_called_once()
            
            # Test format_tools - Cohere has a different format
            tools = [function_to_json(mock_function)]
            formatted_tools = adapter.format_tools(tools)
            # Cohere tools should use a specific format
            self.assertEqual(formatted_tools[0]["name"], "mock_function")
            self.assertIn("parameter_definitions", formatted_tools[0])

if __name__ == '__main__':
    unittest.main()
