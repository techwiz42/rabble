# rabble/adapters/anthropic_adapter.py
import json
import os
from typing import List, Dict, Any, Iterator, Optional

from anthropic import Anthropic

from .base import ModelAdapter

class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""
    
    def __init__(self, client=None, default_model=None, api_key=None):
        """
        Initialize the Anthropic adapter.
        
        Args:
            client: Anthropic client instance
            default_model: Default model to use
            api_key: API key for Anthropic
        """
        # Get API key from parameter, environment, or raise error
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided for Anthropic. Set ANTHROPIC_API_KEY environment variable.")
            
        # Get model from parameter, environment, or raise error
        self.default_model = default_model or os.getenv("ANTHROPIC_DEFAULT_MODEL")
        if not self.default_model:
            raise ValueError("No model specified for Anthropic. Set ANTHROPIC_DEFAULT_MODEL environment variable.")
        
        self.client = client or Anthropic(api_key=self.api_key)
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        model: str = None,
        **kwargs
    ) -> Any:
        """Create a chat completion using the Anthropic API."""
        # Convert messages to Anthropic format if necessary
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
        
        # Filter out system messages as they're handled separately in Anthropic's API
        anthropic_messages = [
            {
                "role": "assistant" if m["role"] == "assistant" else "user",
                "content": m["content"]
            }
            for m in messages if m["role"] != "system"
        ]
        
        create_params = {
            "model": model or self.default_model,
            "messages": anthropic_messages,
            "stream": stream,
            "max_tokens": kwargs.pop("max_tokens", 1000),  # Default to 1000 tokens if not specified
        }
        
        if system_message:
            create_params["system"] = system_message
        
        if tools:
            create_params["tools"] = tools  # Tools are already formatted correctly
        
        # Remove parameters not supported by Anthropic
        kwargs.pop("parallel_tool_calls", None)
        
        # Add remaining kwargs
        for key, value in kwargs.items():
            if key not in create_params:
                create_params[key] = value
        
        return self.client.messages.create(**create_params)
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert standard tool format to Anthropic tool format.
        
        Note: With the updated function_to_json utility, tools should already
        be in the correct format for Anthropic.
        """
        # Tools should already be in the correct format from function_to_json
        return tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from Anthropic completion object."""
        # From API exploration, completion.content is a list of blocks
        content = ""
        tool_calls = []
        
        # Process each content block
        for block in completion.content:
            if hasattr(block, 'type'):
                if block.type == 'text':
                    content += block.text
                elif block.type == 'tool_use':
                    # Found a tool call
                    tool_calls.append({
                        "id": getattr(block, 'id', f"call_{len(tool_calls)}"),
                        "type": "function",
                        "function": {
                            "name": block.tool_use.name,
                            "arguments": json.dumps(block.tool_use.input)
                        }
                    })
        
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls
        }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from an Anthropic stream chunk."""
        result = {}
        
        # Handle different types of stream events from Anthropic
        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
            return {"content": chunk.delta.text}
        elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'tool_use'):
            # Handle tool_use delta
            return {
                "tool_calls": [
                    {
                        "index": 0,  # Anthropic typically returns one tool at a time
                        "function": {
                            "name": chunk.delta.tool_use.name,
                            "arguments": json.dumps(chunk.delta.tool_use.input)
                        }
                    }
                ]
            }
        else:
            return {}
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Anthropic completion."""
        tool_calls = []
        
        # Process each content block looking for tool_use
        if hasattr(completion, 'content'):
            for block in completion.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    tool_calls.append({
                        "id": getattr(block, 'id', f"call_{len(tool_calls)}"),
                        "type": "function",
                        "function": {
                            "name": block.tool_use.name,
                            "arguments": json.dumps(block.tool_use.input)
                        }
                    })
        
        return tool_calls
