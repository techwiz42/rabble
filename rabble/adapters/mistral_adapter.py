# rabble/adapters/mistral_adapter.py
import json
import warnings
import time
import socket
from typing import List, Dict, Any, Iterator, Optional
import os

# Import Mistral API - handle both new (v1.x) and old (v0.x) clients
try:
    # Try the new v1.x client first
    from mistralai import Mistral
    from mistralai.models.sdkerror import SDKError
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
        # Get API key with proper priority: passed param > environment
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided for Mistral. Set MISTRAL_API_KEY environment variable.")
            
        # Get model with proper priority: passed param > environment
        self.default_model = default_model or os.getenv("MISTRAL_DEFAULT_MODEL")
        if not self.default_model:
            raise ValueError("No model specified for Mistral. Set MISTRAL_DEFAULT_MODEL environment variable.")
        
        # Set debug flag
        self.debug = os.getenv("MISTRAL_DEBUG", "0").lower() in ("1", "true", "yes")
        
        # Rate limiting settings
        self.max_retries = int(os.getenv("MISTRAL_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("MISTRAL_RETRY_DELAY", "1.0"))
        
        # Initialize client with proper error handling and diagnostic information
        try:
            if client:
                self.client = client
            elif NEW_CLIENT:
                self.client = Mistral(api_key=self.api_key)
                if self.debug:
                    print(f"Initialized Mistral client v1.x with model: {self.default_model}")
            else:
                from mistralai.client import MistralClient
                self.client = MistralClient(api_key=self.api_key)
                if self.debug:
                    print(f"Initialized Mistral client v0.x with model: {self.default_model}")
        except Exception as e:
            error_msg = f"Mistral client initialization error: {str(e)}"
            if self.debug:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        model: str = None,
        **kwargs
    ) -> Any:
        """Create a chat completion using the Mistral AI API with retry logic."""
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
        
        # Add stream parameter
        create_params["stream"] = stream
        
        # Make the API call with retry logic
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                if self.debug:
                    print(f"Calling Mistral API with model: {create_params['model']}")
                    if stream:
                        print("Streaming mode enabled")
                    if retries > 0:
                        print(f"Retry attempt {retries}/{self.max_retries}")
                
                if NEW_CLIENT:
                    # New client (v1.x)
                    if stream:
                        result = self.client.chat.stream(**create_params)
                        # For streaming, return a special wrapper to handle connection issues
                        return self._create_robust_stream(result) if stream else result
                    else:
                        return self.client.chat.complete(**create_params)
                else:
                    # Old client (v0.x)
                    if stream:
                        result = self.client.chat_stream(**create_params)
                        # For streaming, return a special wrapper to handle connection issues
                        return self._create_robust_stream(result) if stream else result
                    else:
                        return self.client.chat(**create_params)
                
            except Exception as e:
                last_error = e
                # Check if this is a rate limit error
                is_rate_limit = False
                
                if NEW_CLIENT and isinstance(e, SDKError):
                    # Check for 429 status code
                    error_str = str(e)
                    if "429" in error_str or "Requests rate limit exceeded" in error_str:
                        is_rate_limit = True
                else:
                    # Try to identify rate limit errors in other client versions
                    error_str = str(e)
                    if "429" in error_str or "rate limit" in error_str.lower():
                        is_rate_limit = True
                
                # Only retry on rate limit errors
                if is_rate_limit and retries < self.max_retries:
                    wait_time = self.retry_delay * (2 ** retries)  # Exponential backoff
                    if self.debug:
                        print(f"Rate limit exceeded. Waiting {wait_time:.2f}s before retry {retries+1}/{self.max_retries}")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                else:
                    error_msg = f"Mistral API call error: {str(e)}"
                    if self.debug:
                        print(f"ERROR: {error_msg}")
                        print(f"API parameters: model={create_params['model']}, messages={len(create_params['messages'])} items")
                    raise RuntimeError(error_msg)
        
        # If we used all retries
        if last_error:
            error_msg = f"Mistral API call failed after {self.max_retries} retries: {str(last_error)}"
            if self.debug:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _create_robust_stream(self, stream_iterable):
        """
        Create a wrapper around the stream iterator that handles connection errors.
        This is needed because the Mistral Python client has issues with prematurely
        closing connections during streaming.
        
        Args:
            stream_iterable: The original stream iterator
            
        Returns:
            A generator that yields chunks with robust error handling
        """
        def robust_stream():
            try:
                for chunk in stream_iterable:
                    yield chunk
            except (OSError, socket.error, ConnectionError) as e:
                if self.debug:
                    print(f"Stream connection error handled: {str(e)}")
                # Don't re-raise the error, just end the stream gracefully
                return
            except Exception as e:
                if self.debug:
                    print(f"Unexpected error in stream: {str(e)}")
                # For unexpected errors, just end the stream
                return
        
        return robust_stream()
    
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
            if self.debug:
                print(f"Error extracting response from completion: {str(e)}")
            
            # Fallback for any unexpected structure
            try:
                if hasattr(completion, "content"):
                    content = completion.content
                elif hasattr(completion, "choices") and hasattr(completion.choices[0], "message"):
                    content = completion.choices[0].message.content
                else:
                    content = str(completion)
            except Exception:
                content = str(completion)
                
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": []
            }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """
        Extract information from a Mistral AI stream chunk with improved error handling.
        
        Args:
            chunk: A chunk from the Mistral AI streaming response
            
        Returns:
            Dictionary with extracted content and tool calls
        """
        result = {}
        
        try:
            # Extract content - structure differs between client versions
            if NEW_CLIENT:
                # New client (v1.x) - chunk.data.choices[0].delta.content
                if hasattr(chunk, "data") and hasattr(chunk.data, "choices") and chunk.data.choices:
                    delta = chunk.data.choices[0].delta
                    if hasattr(delta, "content") and delta.content is not None:
                        result["content"] = delta.content
            else:
                # Old client (v0.x) - chunk.choices[0].delta.content
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content is not None:
                        result["content"] = delta.content
            
            # Tool calls extraction - with more robust error handling
            try:
                self._try_extract_tool_calls_from_chunk(chunk, result)
            except Exception as e:
                if self.debug:
                    print(f"Error extracting tool calls from stream chunk: {str(e)}")
            
        except (AttributeError, IndexError) as e:
            if self.debug:
                print(f"Error extracting from stream chunk: {str(e)}")
        except (OSError, socket.error, ConnectionError) as e:
            # Handle file descriptor and connection errors
            if self.debug:
                print(f"Stream connection error ignored: {str(e)}")
            # Return the result we have so far rather than raising an exception
        except Exception as e:
            # Catch-all for any other errors
            if self.debug:
                print(f"Unexpected error in stream processing: {str(e)}")
        
        return result
    
    def _try_extract_tool_calls_from_chunk(self, chunk, result):
        """Helper to extract tool calls from stream chunks - with extensive fallbacks."""
        tool_calls = None
        
        if NEW_CLIENT:
            # New client (v1.x)
            if hasattr(chunk, "data") and hasattr(chunk.data, "choices") and chunk.data.choices:
                delta = chunk.data.choices[0].delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_calls = delta.tool_calls
        else:
            # Old client (v0.x)
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_calls = delta.tool_calls
        
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
        except (AttributeError, IndexError) as e:
            if self.debug:
                print(f"Error extracting tool calls: {str(e)}")
        
        return tool_calls
