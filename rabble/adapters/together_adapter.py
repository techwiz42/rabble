# rabble/adapters/together_adapter.py
import json
import os
from typing import List, Dict, Any, Iterator, Optional
import requests

# Import together if available
try:
    import together
    TOGETHER_IMPORTED = True
except ImportError:
    TOGETHER_IMPORTED = False

from .base import ModelAdapter

class TogetherAdapter(ModelAdapter):
    """Adapter for Together AI models."""
    
    def __init__(self, client=None, default_model=None, api_key=None):
        """
        Initialize the Together AI adapter.
        
        Args:
            client: Together client (not used directly)
            default_model: Default model to use
            api_key: API key for Together AI
        """
        self.default_model = os.getenv("TOGETHER_DEFAULT_MODEL")
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
               
        # Together uses module-level API key rather than a client instance
        if TOGETHER_IMPORTED:
            together.api_key = self.api_key
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        model: str = None,
        **kwargs
    ) -> Any:
        """Create a chat completion using the Together AI API."""
        create_params = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": stream,
        }
        
        if tools:
            create_params["tools"] = self.format_tools(tools)
            if tool_choice:
                create_params["tool_choice"] = tool_choice
        
        # Remove parameters not supported by Together
        kwargs.pop("parallel_tool_calls", None)
        
        # Add remaining kwargs
        for key, value in kwargs.items():
            if key not in create_params:
                create_params[key] = value
        
        # Use REST API directly since the together module might not have chat function
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=create_params
        )
        
        if response.status_code != 200:
            error_message = f"Error from Together API: {response.status_code} - {response.text}"
            raise Exception(error_message)
            
        return response.json()
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard tool format to Together AI tool format."""
        # Together AI uses OpenAI-compatible format for tools
        return tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from Together AI completion object."""
        if isinstance(completion, dict):
            # Handle dictionary response format
            content = ""
            tool_calls = []
            
            try:
                if "choices" in completion and completion["choices"]:
                    message = completion["choices"][0]["message"]
                    content = message.get("content", "")
                    
                    # Extract tool calls if present
                    if "tool_calls" in message and message["tool_calls"]:
                        for i, tool_call in enumerate(message["tool_calls"]):
                            tool_calls.append({
                                "id": tool_call.get("id", f"call_{i}"),
                                "type": tool_call.get("type", "function"),
                                "function": {
                                    "name": tool_call["function"]["name"],
                                    "arguments": tool_call["function"]["arguments"]
                                }
                            })
            except (KeyError, IndexError):
                pass
            
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls
            }
        else:
            # Handle object response format (unlikely with our implementation)
            try:
                return {
                    "role": "assistant",
                    "content": completion.choices[0].message.content,
                    "tool_calls": self._extract_tool_calls_from_completion(completion)
                }
            except (AttributeError, IndexError):
                # Fallback for unexpected formats
                return {
                    "role": "assistant",
                    "content": str(completion),
                    "tool_calls": []
                }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from a Together AI stream chunk."""
        result = {}
        
        # Handle different possible chunk formats
        if isinstance(chunk, dict):
            # Dictionary format
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                
                if "content" in delta and delta["content"]:
                    result["content"] = delta["content"]
                
                if "tool_calls" in delta and delta["tool_calls"]:
                    result["tool_calls"] = []
                    for i, tool_call in enumerate(delta["tool_calls"]):
                        if "function" in tool_call:
                            function_data = {
                                "name": tool_call["function"].get("name", ""),
                                "arguments": tool_call["function"].get("arguments", "")
                            }
                            result["tool_calls"].append({
                                "index": i,
                                "function": function_data
                            })
        else:
            # Object format (unlikely with our implementation)
            try:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    if hasattr(delta, "content") and delta.content:
                        result["content"] = delta.content
                    
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        result["tool_calls"] = []
                        for i, tool_call in enumerate(delta.tool_calls):
                            if hasattr(tool_call, "function"):
                                function_data = {
                                    "name": tool_call.function.name if hasattr(tool_call.function, "name") else "",
                                    "arguments": tool_call.function.arguments if hasattr(tool_call.function, "arguments") else ""
                                }
                                result["tool_calls"].append({
                                    "index": i,
                                    "function": function_data
                                })
            except (AttributeError, IndexError):
                pass
        
        return result
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Together AI completion."""
        return self._extract_tool_calls_from_completion(completion)
    
    def _extract_tool_calls_from_completion(self, completion: Any) -> List[Dict[str, Any]]:
        """Helper to extract tool calls from Together AI completion."""
        tool_calls = []
        
        try:
            if isinstance(completion, dict):
                # Handle dictionary format
                if "choices" in completion and completion["choices"]:
                    message = completion["choices"][0]["message"]
                    
                    if "tool_calls" in message and message["tool_calls"]:
                        for i, tool_call in enumerate(message["tool_calls"]):
                            tool_calls.append({
                                "id": tool_call.get("id", f"call_{i}"),
                                "type": tool_call.get("type", "function"),
                                "function": {
                                    "name": tool_call["function"]["name"],
                                    "arguments": tool_call["function"]["arguments"]
                                }
                            })
            else:
                # Handle object format (unlikely with our implementation)
                message = completion.choices[0].message
                
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for i, tool_call in enumerate(message.tool_calls):
                        tool_calls.append({
                            "id": getattr(tool_call, "id", f"call_{i}"),
                            "type": getattr(tool_call, "type", "function"),
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
        except (AttributeError, IndexError, KeyError):
            pass
        
        return tool_calls
