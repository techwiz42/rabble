# rabble/adapters/anthropic_adapter.py
import json
from typing import List, Dict, Any, Iterator, Optional

from anthropic import Anthropic

from .base import ModelAdapter

class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""
    
    def __init__(self, client=None, default_model="claude-3-5-sonnet-20240620"):
        """
        Initialize the Anthropic adapter.
        
        Args:
            client: Anthropic client instance
            default_model: Default model to use
        """
        self.client = client or Anthropic()
        self.default_model = default_model
    
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
        # Convert messages to Anthropic format
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
        }
        
        if system_message:
            create_params["system"] = system_message
        
        if tools:
            create_params["tools"] = self.format_tools(tools)
            # Anthropic doesn't have a direct equivalent of tool_choice,
            # but we can still pass it for future compatibility
        
        # Remove parallel_tool_calls if present as it's not supported by Anthropic
        kwargs.pop("parallel_tool_calls", None)
        
        # Add remaining kwargs
        for key, value in kwargs.items():
            if key not in create_params:
                create_params[key] = value
        
        return self.client.messages.create(**create_params)
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard tool format to Anthropic tool format."""
        anthropic_tools = []
        
        for tool in tools:
            if tool["type"] == "function":
                function_def = tool["function"]
                anthropic_tool = {
                    "name": function_def["name"],
                    "description": function_def.get("description", ""),
                    "input_schema": {
                        "type": "object",
                        "properties": function_def["parameters"].get("properties", {}),
                        "required": function_def["parameters"].get("required", [])
                    }
                }
                anthropic_tools.append({"function": anthropic_tool})
        
        return anthropic_tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from Anthropic completion object."""
        return {
            "role": "assistant",
            "content": completion.content[0].text if completion.content else "",
            "tool_calls": self._extract_tool_calls_from_completion(completion)
        }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from an Anthropic stream chunk."""
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
                            "arguments": chunk.delta.tool_use.input
                        }
                    }
                ]
            }
        else:
            return {}
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Anthropic completion."""
        return self._extract_tool_calls_from_completion(completion)
    
    def _extract_tool_calls_from_completion(self, completion: Any) -> List[Dict[str, Any]]:
        """Helper to extract tool calls from various Anthropic response formats."""
        tool_calls = []
        
        # Check for tool_use in content blocks
        for content_block in getattr(completion, 'content', []):
            if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",  # Generate an ID since Anthropic doesn't provide one
                    "type": "function",
                    "function": {
                        "name": content_block.tool_use.name,
                        "arguments": json.dumps(content_block.tool_use.input)
                    }
                })
        
        return tool_calls
