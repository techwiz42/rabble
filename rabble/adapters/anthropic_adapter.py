# rabble/adapters/anthropic_adapter.py
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Import Anthropic SDK
from anthropic import Anthropic

from .base import ModelAdapter

class AnthropicAdapter(ModelAdapter):
    """
    Adapter for Anthropic Claude models.
    
    This adapter handles the conversion between a standardized API format and
    Anthropic's specific API requirements, with special attention to properly
    handling Anthropic's unique tool usage format.
    """
    
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
        
        # Initialize the client
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
        """
        Create a chat completion using the Anthropic API.
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            tool_choice: Optional specification for which tool to use
            stream: Whether to stream the response
            model: Model to use, overrides the default
            **kwargs: Additional parameters to pass to the Anthropic API
            
        Returns:
            Anthropic API response
        """
        # Extract system message and convert messages to Anthropic format
        system_message, anthropic_messages = self._convert_messages_to_anthropic_format(messages)
        
        # Prepare API request parameters
        create_params = {
            "model": model or self.default_model,
            "messages": anthropic_messages,
            "stream": stream,
            "max_tokens": kwargs.pop("max_tokens", 1024),  # Default to 1024 tokens if not specified
        }
        
        # Add system message if present
        if system_message:
            create_params["system"] = system_message
        
        # Handle tools if provided
        if tools:
            # Convert tools to Anthropic's format
            create_params["tools"] = self.format_tools(tools)
            
            # Handle tool_choice if specified
            if tool_choice:
                create_params["tool_choice"] = self._convert_tool_choice(tool_choice)
        
        # Remove parameters not supported by Anthropic
        kwargs.pop("parallel_tool_calls", None)
        
        # Add remaining kwargs
        for key, value in kwargs.items():
            if key not in create_params:
                create_params[key] = value
        
        # Make the API call with appropriate error handling
        try:
            return self.client.messages.create(**create_params)
        except Exception as e:
            error_msg = f"Anthropic API call failed: {str(e)}"
            # Add details about the request for debugging
            if model:
                error_msg += f" (model: {model})"
            raise RuntimeError(error_msg)
    
    def _convert_messages_to_anthropic_format(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert standard message format to Anthropic's format.
        
        Args:
            messages: List of message dictionaries in standard format
            
        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        # For debugging
        if os.getenv("ANTHROPIC_DEBUG"):
            print("Original messages:")
            for i, msg in enumerate(messages):
                print(f"  {i}: {msg.get('role')}: {msg.get('content', '')[:50]}... {msg.get('tool_calls', [])}")
        # Extract system message if present
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
        
        # Build a tool call lookup dict for matching tool results with their calls
        tool_call_lookup = {}
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("id"):  # Ensure there's an ID
                        tool_call_lookup[tool_call.get("id")] = {
                            "index": i,
                            "name": tool_call.get("function", {}).get("name", ""),
                            "arguments": tool_call.get("function", {}).get("arguments", "{}")
                        }
        
        # Process messages to Anthropic format
        anthropic_messages = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # Skip system messages as they're handled separately
            if msg["role"] == "system":
                i += 1
                continue
            
            if msg["role"] == "user" or msg["role"] == "assistant":
                # Handle regular user or assistant messages
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    # Format assistant message with tool calls
                    content_blocks = []
                    
                    # Add text content if present
                    if msg.get("content"):
                        content_blocks.append({"type": "text", "text": msg["content"]})
                    
                    # Add tool_use blocks for each tool call
                    for tool_call in msg.get("tool_calls", []):
                        try:
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tool_call.get("id", ""),
                                "name": tool_call.get("function", {}).get("name", ""),
                                "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                            })
                        except json.JSONDecodeError:
                            # Handle case where arguments isn't valid JSON
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tool_call.get("id", ""),
                                "name": tool_call.get("function", {}).get("name", ""),
                                "input": {}  # Default to empty object if JSON parsing fails
                            })
                    
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
                    
                    # Look ahead for any tool results
                    next_index = i + 1
                    while next_index < len(messages) and messages[next_index].get("role") == "tool":
                        tool_result_msg = messages[next_index]
                        tool_call_id = tool_result_msg.get("tool_call_id")
                        
                        # Create tool result message
                        anthropic_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_call_id,
                                    "content": tool_result_msg.get("content", "")
                                }
                            ]
                        })
                        
                        next_index += 1
                    
                    # Skip any tool messages we've processed
                    i = next_index
                else:
                    # Regular message without tool calls
                    anthropic_messages.append({
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"]
                    })
                    i += 1
            elif msg["role"] == "tool":
                # Handle standalone tool results (not preceded by assistant)
                # This is a fallback case and might not occur in well-formed conversations
                tool_call_id = msg.get("tool_call_id")
                
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": msg.get("content", "")
                        }
                    ]
                })
                i += 1
            else:
                # Skip unknown message types
                i += 1
        
        # For debugging
        if os.getenv("ANTHROPIC_DEBUG"):
            print("\nConverted Anthropic messages:")
            for i, msg in enumerate(anthropic_messages):
                if isinstance(msg.get("content"), list):
                    content_desc = f"{len(msg['content'])} content blocks"
                    for block in msg["content"]:
                        if block.get("type") == "tool_result":
                            content_desc += f" (tool_result for {block.get('tool_use_id', 'unknown')})"
                        elif block.get("type") == "tool_use":
                            content_desc += f" (tool_use {block.get('name', 'unknown')})"
                else:
                    content_desc = f"{msg.get('content', '')[:50]}..."
                print(f"  {i}: {msg.get('role')}: {content_desc}")
        
        return system_message, anthropic_messages
    
    def _convert_tool_choice(self, tool_choice: str) -> Dict[str, Any]:
        """
        Convert standardized tool_choice to Anthropic format.
        
        Args:
            tool_choice: Tool choice specification
            
        Returns:
            Tool choice in Anthropic format
        """
        if tool_choice == "none":
            return {"type": "none"}
        elif tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "any":
            return {"type": "any"}
        else:
            # Assume it's a specific tool name
            return {
                "type": "tool",
                "name": tool_choice
            }
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert standard tool format to Anthropic tool format.
        
        Args:
            tools: List of tools in standard format
            
        Returns:
            List of tools in Anthropic format
        """
        anthropic_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                function_def = tool.get("function", {})
                
                # Convert to Anthropic's tool format
                anthropic_tool = {
                    "name": function_def.get("name", ""),
                    "description": function_def.get("description", ""),
                    "input_schema": function_def.get("parameters", {})
                }
                
                # Ensure input_schema has the right structure
                if "input_schema" in anthropic_tool and isinstance(anthropic_tool["input_schema"], dict):
                    # Ensure type is object at the top level if not already specified
                    if "type" not in anthropic_tool["input_schema"]:
                        anthropic_tool["input_schema"]["type"] = "object"
                
                anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """
        Extract response from Anthropic completion object.
        
        Args:
            completion: Anthropic API response
            
        Returns:
            Standardized response dictionary
        """
        content = ""
        tool_calls = []
        
        # Process each content block
        if hasattr(completion, "content"):
            for block in completion.content:
                if hasattr(block, 'type'):
                    if block.type == 'text':
                        content += block.text
                    elif block.type == 'tool_use':
                        # Found a tool call
                        try:
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input)
                                }
                            })
                        except Exception as e:
                            # Handle unexpected tool_use format
                            tool_calls.append({
                                "id": getattr(block, "id", f"call_{len(tool_calls)}"),
                                "type": "function",
                                "function": {
                                    "name": getattr(block, "name", "unknown"),
                                    "arguments": "{}"
                                }
                            })
        
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls if tool_calls else None
        }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """
        Extract information from an Anthropic stream chunk.
        
        Args:
            chunk: Anthropic stream event
            
        Returns:
            Standardized chunk dictionary
        """
        result = {}
        
        try:
            # Handle different types of stream events from Anthropic
            if hasattr(chunk, 'type'):
                # Content block delta - most common for text
                if chunk.type == 'content_block_delta':
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'type'):
                        # Text delta
                        if chunk.delta.type == 'text_delta' and hasattr(chunk.delta, 'text'):
                            result["content"] = chunk.delta.text
                        # Tool use delta
                        elif chunk.delta.type == 'tool_use_delta':
                            tool_call = {
                                "index": 0,  # Anthropic typically returns one tool at a time
                                "function": {}
                            }
                            
                            if hasattr(chunk.delta, 'name'):
                                tool_call["function"]["name"] = chunk.delta.name
                            if hasattr(chunk.delta, 'input') and chunk.delta.input:
                                tool_call["function"]["arguments"] = json.dumps(chunk.delta.input)
                            
                            # Only add tool_calls if we have meaningful data
                            if tool_call["function"].get("name") or tool_call["function"].get("arguments"):
                                result["tool_calls"] = [tool_call]
                
                # Complete tool use in a single block
                elif chunk.type == 'content_block_start':
                    if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'type'):
                        # Text block
                        if chunk.content_block.type == 'text' and hasattr(chunk.content_block, 'text'):
                            result["content"] = chunk.content_block.text
                        # Tool use block
                        elif chunk.content_block.type == 'tool_use':
                            # Get tool details
                            name = getattr(chunk.content_block, 'name', '')
                            tool_input = getattr(chunk.content_block, 'input', {})
                            tool_id = getattr(chunk.content_block, 'id', '')
                            
                            # Only add if we have a name
                            if name:
                                result["tool_calls"] = [
                                    {
                                        "index": 0,
                                        "id": tool_id,
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(tool_input)
                                        }
                                    }
                                ]
                
                # Message delta - might contain stop reason
                elif chunk.type == 'message_delta':
                    # Nothing to extract for the standard streaming interface
                    pass
                
                # Message stop
                elif chunk.type == 'message_stop':
                    # Could add a special marker if needed
                    pass
        except Exception as e:
            # Silent error handling for stream extraction
            # Don't raise exceptions during streaming as it would break the stream
            pass
        
        return result
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Anthropic completion.
        
        Args:
            completion: Anthropic API response
            
        Returns:
            List of standardized tool call dictionaries
        """
        tool_calls = []
        
        try:
            # Process each content block looking for tool_use
            if hasattr(completion, 'content'):
                for i, block in enumerate(completion.content):
                    if hasattr(block, 'type') and block.type == 'tool_use':
                        tool_calls.append({
                            "id": getattr(block, 'id', f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": getattr(block, 'name', ''),
                                "arguments": json.dumps(getattr(block, 'input', {}))
                            }
                        })
        except Exception as e:
            # Handle unexpected formats or missing attributes
            pass
        
        return tool_calls
