# rabble/adapters/cohere_adapter.py
import json
from typing import List, Dict, Any, Iterator, Optional

import cohere

from .base import ModelAdapter

class CohereAdapter(ModelAdapter):
    """Adapter for Cohere models."""
    
    def __init__(self, client=None, default_model="command", api_key=None):
        """
        Initialize the Cohere adapter.
        
        Args:
            client: Cohere client instance
            default_model: Default model to use
            api_key: Optional API key if client is not provided
        """
        self.client = client or cohere.Client(api_key=api_key)
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
        """Create a chat completion using the Cohere API."""
        # Extract system message if present
        system_message = None
        chat_history = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user" or msg["role"] == "assistant":
                chat_history.append({
                    "role": msg["role"],
                    "message": msg["content"]
                })
        
        create_params = {
            "model": model or self.default_model,
            "message": chat_history[-1]["message"] if chat_history else "",
            "chat_history": chat_history[:-1] if len(chat_history) > 1 else None,
        }
        
        # Add system message if present
        if system_message:
            create_params["preamble"] = system_message
        
        if tools:
            create_params["tools"] = self.format_tools(tools)
        
        # Remove parameters not supported by Cohere
        kwargs.pop("parallel_tool_calls", None)
        kwargs.pop("tool_choice", None)
        
        # Remove stream parameter as it's not supported in Cohere's chat method
        kwargs.pop("stream", None)
        
        # Handle max_tokens (Cohere uses max_tokens)
        if "max_tokens" in kwargs:
            create_params["max_tokens"] = kwargs.pop("max_tokens")
        
        # Add remaining kwargs
        for key, value in kwargs.items():
            if key not in create_params:
                create_params[key] = value
        
        # Note: Cohere doesn't support streaming in the same way as OpenAI
        return self.client.chat(**create_params)
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard tool format to Cohere tool format."""
        cohere_tools = []
        
        for tool in tools:
            if tool["type"] == "function":
                function_def = tool["function"]
                cohere_tool = {
                    "name": function_def["name"],
                    "description": function_def.get("description", ""),
                    "parameter_definitions": {}
                }
                
                # Convert parameters from OpenAI format to Cohere
                if "parameters" in function_def and "properties" in function_def["parameters"]:
                    props = function_def["parameters"]["properties"]
                    required = function_def["parameters"].get("required", [])
                    
                    for param_name, param_def in props.items():
                        param_type = param_def.get("type", "string")
                        cohere_tool["parameter_definitions"][param_name] = {
                            "description": param_def.get("description", ""),
                            "type": param_type,
                            "required": param_name in required
                        }
                
                cohere_tools.append(cohere_tool)
        
        return cohere_tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from Cohere completion object."""
        try:
            return {
                "role": "assistant",
                "content": completion.text,
                "tool_calls": self._extract_tool_calls_from_completion(completion)
            }
        except AttributeError:
            # Handle dictionary responses
            if isinstance(completion, dict):
                return {
                    "role": "assistant",
                    "content": completion.get("text", ""),
                    "tool_calls": self._extract_tool_calls_from_completion(completion)
                }
            else:
                # Ultimate fallback
                return {
                    "role": "assistant",
                    "content": str(completion),
                    "tool_calls": []
                }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from a Cohere stream chunk."""
        # Note: Cohere doesn't support streaming in the same way as OpenAI
        # This is a placeholder for compatibility
        result = {}
        
        if hasattr(chunk, "text") and chunk.text:
            result["content"] = chunk.text
        elif isinstance(chunk, dict) and "text" in chunk:
            result["content"] = chunk["text"]
        
        # Extract tool calls if available (they come complete, not streamed)
        tool_calls = self._extract_tool_calls_from_completion(chunk)
        if tool_calls:
            result["tool_calls"] = []
            for i, tool_call in enumerate(tool_calls):
                result["tool_calls"].append({
                    "index": i,
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"]
                    }
                })
        
        return result
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Cohere completion."""
        return self._extract_tool_calls_from_completion(completion)
    
    def _extract_tool_calls_from_completion(self, completion: Any) -> List[Dict[str, Any]]:
        """Helper to extract tool calls from Cohere completion."""
        tool_calls = []
        
        try:
            # Try object attribute access first
            if hasattr(completion, "tool_calls") and completion.tool_calls:
                for i, tool_call in enumerate(completion.tool_calls):
                    # Convert Cohere tool call format to standard format
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": getattr(tool_call, "name", ""),
                            "arguments": json.dumps(getattr(tool_call, "parameters", {}))
                        }
                    })
            # Try dictionary access next
            elif isinstance(completion, dict) and "tool_calls" in completion:
                for i, tool_call in enumerate(completion["tool_calls"]):
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": json.dumps(tool_call.get("parameters", {}))
                        }
                    })
        except (AttributeError, TypeError):
            pass
        
        return tool_calls
