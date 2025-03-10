# rabble/adapters/deepseek_adapter.py
import json
from typing import List, Dict, Any, Iterator, Optional

from deepseek import DeepSeek

from .base import ModelAdapter

class DeepSeekAdapter(ModelAdapter):
    """Adapter for DeepSeek models."""
    
    def __init__(self, client=None, default_model="deepseek-chat"):
        """
        Initialize the DeepSeek adapter.
        
        Args:
            client: DeepSeek client instance
            default_model: Default model to use
        """
        self.client = client or DeepSeek()
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
        """Create a chat completion using the DeepSeek API."""
        create_params = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": stream,
        }
        
        if tools:
            create_params["tools"] = self.format_tools(tools)
            if tool_choice:
                # Different providers handle tool_choice differently, adapt as needed
                create_params["tool_choice"] = "auto"  # DeepSeek might use different values
        
        # Remove parameters not supported by DeepSeek
        kwargs.pop("parallel_tool_calls", None)
        
        # Add remaining kwargs
        for key, value in kwargs.items():
            if key not in create_params:
                create_params[key] = value
        
        return self.client.chat.completions.create(**create_params)
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard tool format to DeepSeek tool format."""
        # For this example, assuming DeepSeek uses a format similar to OpenAI
        # Adjust based on actual DeepSeek API documentation
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
        # Adapt based on actual DeepSeek API response format
        message = completion.choices[0].message
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": self._extract_tool_calls_from_message(message)
        }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from a DeepSeek stream chunk."""
        # Adapt based on actual DeepSeek API stream format
        delta = chunk.choices[0].delta
        result = {}
        
        if hasattr(delta, "content") and delta.content:
            result["content"] = delta.content
        
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            # Adapt based on how DeepSeek formats tool calls in stream chunks
            tool_call = delta.tool_calls[0]
            result["tool_calls"] = [{
                "index": tool_call.index,
                "function": {
                    "name": tool_call.function.name if hasattr(tool_call.function, "name") else "",
                    "arguments": tool_call.function.arguments if hasattr(tool_call.function, "arguments") else ""
                }
            }]
        
        return result
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from DeepSeek completion."""
        message = completion.choices[0].message
        return self._extract_tool_calls_from_message(message)
    
    def _extract_tool_calls_from_message(self, message: Any) -> List[Dict[str, Any]]:
        """Helper to extract tool calls from a DeepSeek message."""
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []
        
        tool_calls = []
        for tool_call in message.tool_calls:
            tool_calls.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
        
        return tool_calls
