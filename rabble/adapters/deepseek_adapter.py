# rabble/adapters/deepseek_adapter.py
import json
import os
from typing import List, Dict, Any, Iterator, Optional

from .base import ModelAdapter

class DeepSeekAdapter(ModelAdapter):
    """Adapter for DeepSeek models."""
    
    def __init__(self, client=None, default_model=None, api_key=None):
        """
        Initialize the DeepSeek adapter.
        
        Args:
            client: DeepSeekAPI client instance
            default_model: Default model to use
            api_key: API key for DeepSeek
        """
        # Get API key from parameter, environment, or raise error
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided for DeepSeek. Set DEEPSEEK_API_KEY environment variable.")
            
        # Get model from parameter, environment, or raise error
        self.default_model = default_model or os.getenv("DEEPSEEK_DEFAULT_MODEL")
        if not self.default_model:
            raise ValueError("No model specified for DeepSeek. Set DEEPSEEK_DEFAULT_MODEL environment variable.")
        
        # Initialize the client if not provided
        if client:
            self.client = client
        else:
            try:
                from deepseek import DeepSeekAPI
                self.client = DeepSeekAPI(api_key=self.api_key)
            except ImportError:
                raise ImportError("DeepSeek SDK not installed. Run 'pip install deepseek'.")
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        model: str = None,
        **kwargs
    ) -> Any:
        """Create a chat completion using the DeepSeek API."""
        create_params = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": stream,
        }
        
        if tools:
            create_params["tools"] = self.format_tools(tools)
        
        # Remove parameters not supported by DeepSeek
        kwargs.pop("parallel_tool_calls", None)
        
        # Add remaining kwargs
        for key, value in kwargs.items():
            if key not in create_params:
                create_params[key] = value
        
        return self.client.chat_completion(**create_params)
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard tool format to DeepSeek tool format."""
        deepseek_tools = []
        
        for tool in tools:
            if tool["type"] == "function":
                deepseek_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"]["parameters"]
                    }
                })
        
        return deepseek_tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from DeepSeek completion object."""
        if isinstance(completion, dict):
            # Handle the dict response from DeepSeek API
            message = completion.get("choices", [{}])[0].get("message", {})
            return {
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": self._extract_tool_calls_from_completion(completion)
            }
        else:
            # If it's not a dict, try to access attributes
            try:
                message = completion.choices[0].message
                return {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": self._extract_tool_calls_from_message(message)
                }
            except (AttributeError, IndexError):
                # Fallback for unexpected response format
                return {
                    "role": "assistant",
                    "content": str(completion),
                    "tool_calls": []
                }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from a DeepSeek stream chunk."""
        result = {}
        
        # Handle different possible response formats from DeepSeek
        if isinstance(chunk, dict):
            # Dict format
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                result["content"] = delta["content"]
            
            if "tool_calls" in delta and delta["tool_calls"]:
                # Process tool calls
                tool_call = delta["tool_calls"][0]
                result["tool_calls"] = [{
                    "index": 0,  # Default index if not provided
                    "function": {
                        "name": tool_call.get("function", {}).get("name", ""),
                        "arguments": tool_call.get("function", {}).get("arguments", "")
                    }
                }]
        else:
            # Try to handle object format
            try:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        result["content"] = delta.content
                    
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_call = delta.tool_calls[0]
                        result["tool_calls"] = [{
                            "index": getattr(tool_call, "index", 0),
                            "function": {
                                "name": getattr(tool_call.function, "name", ""),
                                "arguments": getattr(tool_call.function, "arguments", "")
                            }
                        }]
            except (AttributeError, IndexError):
                # Fallback for unexpected format
                pass
        
        return result
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from DeepSeek completion."""
        return self._extract_tool_calls_from_completion(completion)
    
    def _extract_tool_calls_from_completion(self, completion: Any) -> List[Dict[str, Any]]:
        """Helper to extract tool calls from a DeepSeek completion."""
        tool_calls = []
        
        if isinstance(completion, dict):
            # Dict format
            message = completion.get("choices", [{}])[0].get("message", {})
            tool_calls_data = message.get("tool_calls", [])
            
            for i, tool_call in enumerate(tool_calls_data):
                tool_calls.append({
                    "id": tool_call.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("function", {}).get("name", ""),
                        "arguments": tool_call.get("function", {}).get("arguments", "{}")
                    }
                })
        else:
            # Try to handle object format
            try:
                message = completion.choices[0].message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for i, tool_call in enumerate(message.tool_calls):
                        tool_calls.append({
                            "id": getattr(tool_call, "id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": getattr(tool_call.function, "name", ""),
                                "arguments": getattr(tool_call.function, "arguments", "{}")
                            }
                        })
            except (AttributeError, IndexError):
                # No tool calls or unexpected format
                pass
        
        return tool_calls
    
    def _extract_tool_calls_from_message(self, message: Any) -> List[Dict[str, Any]]:
        """Helper to extract tool calls from a DeepSeek message."""
        tool_calls = []
        
        try:
            if hasattr(message, "tool_calls") and message.tool_calls:
                for i, tool_call in enumerate(message.tool_calls):
                    tool_calls.append({
                        "id": getattr(tool_call, "id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": getattr(tool_call.function, "name", ""),
                            "arguments": getattr(tool_call.function, "arguments", "{}")
                        }
                    })
        except AttributeError:
            # No tool calls or unexpected format
            pass
        
        return tool_calls
