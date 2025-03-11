# rabble/adapters/mistral_adapter.py
import json
import warnings
from typing import List, Dict, Any, Iterator, Optional
import os

# Import Mistral API - handle both new (v1.x) and old (v0.x) clients
try:
    # Try the new v1.x client first
    from mistralai import Mistral
    NEW_CLIENT = True
except ImportError:
    # Fall back to deprecated v0.x client with a warning
    warnings.warn("Using deprecated Mistral client. Consider upgrading to latest mistralai>=1.0.0")
    try:
        from mistralai.client import MistralClient
        NEW_CLIENT = False
    except ImportError:
        # If neither works, imports will fail at runtime
        NEW_CLIENT = False

from .base import ModelAdapter

class MistralAdapter(ModelAdapter):
    """Adapter for Mistral AI models."""
    
    def __init__(self, client=None, default_model=None, api_key=None):
        """
        Initialize the Mistral AI adapter.
        
        Args:
            client: Mistral client instance
            default_model: Default model to use
            api_key: Optional API key if client is not provided
        """
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.default_model = os.getenv("MISTRAL_DEFAULT_MODEL")
        
        if client:
            self.client = client
        elif NEW_CLIENT:
            self.client = Mistral(api_key=api_key)
        else:
            from mistralai.client import MistralClient
            self.client = MistralClient(api_key=api_key)
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        model: str = None,
        **kwargs
    ) -> Any:
        """Create a chat completion using the Mistral AI API."""
        create_params = {
            "model": model or self.default_model,
            "messages": messages,
        }
        
        # Handle max_tokens
        if "max_tokens" in kwargs:
            create_params["max_tokens"] = kwargs.pop("max_tokens")
        
        # Add tools if supported
        if tools:
            create_params["tools"] = self.format_tools(tools)
            if tool_choice:
                create_params["tool_choice"] = tool_choice
        
        # Add remaining parameters
        for key, value in kwargs.items():
            if key not in create_params and key not in ["parallel_tool_calls"]:
                create_params[key] = value
        
        # Make the API call based on client version
        if NEW_CLIENT:
            # New client (v1.x)
            if stream:
                return self.client.chat.stream(**create_params)
            else:
                return self.client.chat.complete(**create_params)
        else:
            # Old client (v0.x)
            if stream:
                return self.client.chat_stream(**create_params)
            else:
                return self.client.chat(**create_params)
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard tool format to Mistral AI tool format."""
        # Both client versions use OpenAI-compatible tools format
        return tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from Mistral AI completion object."""
        try:
            # Structure differs between client versions
            if NEW_CLIENT:
                # New client (v1.x) response format
                return {
                    "role": "assistant",
                    "content": completion.choices[0].message.content,
                    "tool_calls": self._extract_tool_calls_from_completion(completion)
                }
            else:
                # Old client (v0.x) response format
                return {
                    "role": "assistant",
                    "content": completion.choices[0].message.content,
                    "tool_calls": self._extract_tool_calls_from_completion(completion)
                }
        except (AttributeError, IndexError) as e:
            # Fallback for any unexpected structure
            try:
                if hasattr(completion, "content"):
                    content = completion.content
                elif hasattr(completion, "choices") and hasattr(completion.choices[0], "message"):
                    content = completion.choices[0].message.content
                else:
                    content = str(completion)
            except:
                content = str(completion)
                
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": []
            }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from a Mistral AI stream chunk."""
        result = {}
        
        try:
            # Extract content - structure differs between client versions
            if NEW_CLIENT:
                # New client (v1.x) - chunk.data.choices[0].delta.content
                if hasattr(chunk, "data") and hasattr(chunk.data, "choices") and chunk.data.choices:
                    if hasattr(chunk.data.choices[0], "delta") and hasattr(chunk.data.choices[0].delta, "content"):
                        if chunk.data.choices[0].delta.content is not None:
                            result["content"] = chunk.data.choices[0].delta.content
            else:
                # Old client (v0.x) - chunk.choices[0].delta.content
                if hasattr(chunk, "choices") and chunk.choices:
                    if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                        if chunk.choices[0].delta.content is not None:
                            result["content"] = chunk.choices[0].delta.content
            
            # Tool calls extraction - much more limited in the stream
            self._try_extract_tool_calls_from_chunk(chunk, result)
        except Exception:
            # Silent error handling for stream extraction
            pass
        
        return result
    
    def _try_extract_tool_calls_from_chunk(self, chunk, result):
        """Helper to extract tool calls from stream chunks - with extensive fallbacks."""
        try:
            tool_calls = None
            
            if NEW_CLIENT:
                # New client (v1.x)
                if hasattr(chunk, "data") and hasattr(chunk.data, "choices") and chunk.data.choices:
                    if hasattr(chunk.data.choices[0], "delta") and hasattr(chunk.data.choices[0].delta, "tool_calls"):
                        tool_calls = chunk.data.choices[0].delta.tool_calls
            else:
                # Old client (v0.x)
                if hasattr(chunk, "choices") and chunk.choices:
                    if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "tool_calls"):
                        tool_calls = chunk.choices[0].delta.tool_calls
            
            if tool_calls:
                result["tool_calls"] = []
                for i, tool_call in enumerate(tool_calls):
                    tool_call_data = {"index": i, "function": {}}
                    
                    # Extract function info with fallbacks
                    if hasattr(tool_call, "function"):
                        if hasattr(tool_call.function, "name"):
                            tool_call_data["function"]["name"] = tool_call.function.name
                            
                        if hasattr(tool_call.function, "arguments"):
                            tool_call_data["function"]["arguments"] = tool_call.function.arguments
                            
                    result["tool_calls"].append(tool_call_data)
        except Exception:
            # Silent error handling
            pass
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Mistral AI completion."""
        return self._extract_tool_calls_from_completion(completion)
    
    def _extract_tool_calls_from_completion(self, completion: Any) -> List[Dict[str, Any]]:
        """Helper to extract tool calls from Mistral AI completion."""
        tool_calls = []
        
        try:
            # Structure differs between client versions
            message = None
            
            if NEW_CLIENT:
                # New client (v1.x)
                if hasattr(completion, "choices") and completion.choices:
                    message = completion.choices[0].message
            else:
                # Old client (v0.x)
                if hasattr(completion, "choices") and completion.choices:
                    message = completion.choices[0].message
            
            if message and hasattr(message, "tool_calls") and message.tool_calls:
                for i, tool_call in enumerate(message.tool_calls):
                    tool_call_data = {
                        "id": getattr(tool_call, "id", f"call_{i}"),
                        "type": "function",
                        "function": {}
                    }
                    
                    if hasattr(tool_call, "function"):
                        if hasattr(tool_call.function, "name"):
                            tool_call_data["function"]["name"] = tool_call.function.name
                        
                        if hasattr(tool_call.function, "arguments"):
                            tool_call_data["function"]["arguments"] = tool_call.function.arguments
                            
                    tool_calls.append(tool_call_data)
        except (AttributeError, IndexError):
            pass
        
        return tool_calls
