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
        user_and_assistant_messages = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                continue
                
            if message["role"] == "tool":
                # Find the preceding assistant message that used this tool
                tool_call_id = message.get("tool_call_id")
                for j in range(i-1, -1, -1):
                    prev_msg = messages[j]
                    if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                        # Found the assistant message that called this tool
                        matching_tool_call = next(
                            (tc for tc in prev_msg.get("tool_calls", []) 
                             if tc.get("id") == tool_call_id),
                            None
                        )
                        if matching_tool_call:
                            # Convert to Anthropic's tool_result format
                            user_and_assistant_messages.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_call_id": tool_call_id,
                                        "content": message.get("content", "")
                                    }
                                ]
                            })
                        break
            else:
                # Regular user or assistant message
                if message["role"] == "assistant" and message.get("tool_calls"):
                    # Convert assistant message with tool_calls to Anthropic format
                    content_blocks = []
                    
                    # Add text content if present
                    if message.get("content"):
                        content_blocks.append({"type": "text", "text": message["content"]})
                    
                    # Add tool_use blocks
                    for tool_call in message.get("tool_calls", []):
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_call.get("id", ""),
                            "name": tool_call.get("function", {}).get("name", ""),
                            "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                        })
                    
                    user_and_assistant_messages.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
                else:
                    # Regular message without tool calls
                    user_and_assistant_messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
        
        # Prepare API request parameters
        create_params = {
            "model": model or self.default_model,
            "messages": user_and_assistant_messages,
            "stream": stream,
            "max_tokens": kwargs.pop("max_tokens", 1000),  # Default to 1000 tokens if not specified
        }
        
        if system_message:
            create_params["system"] = system_message
        
        if tools:
            # Convert tools to Anthropic's format
            anthropic_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    function_def = tool.get("function", {})
                    anthropic_tools.append({
                        "name": function_def.get("name", ""),
                        "description": function_def.get("description", ""),
                        "input_schema": function_def.get("parameters", {})
                    })
            create_params["tools"] = anthropic_tools
            
            # Handle tool_choice if specified
            if tool_choice:
                if tool_choice == "none":
                    create_params["tool_choice"] = {"type": "none"}
                elif tool_choice == "auto":
                    create_params["tool_choice"] = {"type": "auto"}
                elif tool_choice == "any":
                    create_params["tool_choice"] = {"type": "any"}
                else:
                    # Specific tool choice
                    create_params["tool_choice"] = {
                        "type": "tool",
                        "name": tool_choice
                    }
        
        # Remove parameters not supported by Anthropic
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
            if tool.get("type") == "function":
                function_def = tool.get("function", {})
                anthropic_tools.append({
                    "name": function_def.get("name", ""),
                    "description": function_def.get("description", ""),
                    "input_schema": function_def.get("parameters", {})
                })
        
        return anthropic_tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from Anthropic completion object."""
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
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })
        
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls if tool_calls else None
        }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from an Anthropic stream chunk."""
        result = {}
        
        # Handle different types of stream events from Anthropic
        if hasattr(chunk, 'type'):
            if chunk.type == 'content_block_delta':
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'type'):
                    if chunk.delta.type == 'text_delta':
                        return {"content": chunk.delta.text}
                    elif chunk.delta.type == 'tool_use_delta':
                        # Handle tool_use delta
                        tool_call = {
                            "index": 0,  # Anthropic typically returns one tool at a time
                            "function": {}
                        }
                        
                        if hasattr(chunk.delta, 'name'):
                            tool_call["function"]["name"] = chunk.delta.name
                        if hasattr(chunk.delta, 'input'):
                            tool_call["function"]["arguments"] = json.dumps(chunk.delta.input)
                        
                        return {"tool_calls": [tool_call]}
            
            # For initial content block
            elif chunk.type == 'content_block_start':
                if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'type'):
                    if chunk.content_block.type == 'tool_use':
                        # Initial tool use block
                        return {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": chunk.content_block.name,
                                        "arguments": json.dumps(chunk.content_block.input)
                                    }
                                }
                            ]
                        }
        
        return result
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Anthropic completion."""
        tool_calls = []
        
        # Process each content block looking for tool_use
        if hasattr(completion, 'content'):
            for block in completion.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })
        
        return tool_calls
