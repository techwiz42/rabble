# rabble/adapters/google_adapter.py
import json
import os
from typing import List, Dict, Any, Iterator, Optional

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration

from .base import ModelAdapter

class GoogleAdapter(ModelAdapter):
    """Adapter for Google Gemini models."""
    
    def __init__(self, client=None, default_model=None, api_key=None):
        """
        Initialize the Google Gemini adapter.
        
        Args:
            client: Google GenerativeAI client (not used directly)
            default_model: Default model to use
            api_key: API key for Google Gemini
        """
        self.default_model = os.getenv("GOOGLE_DEFAULT_MODEL")
        
        # Get API key from parameter, environment, or raise error
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided for Google Gemini. Set GOOGLE_API_KEY environment variable.")
        
        # Configure the genai module with the API key
        genai.configure(api_key=self.api_key)
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        model: str = None,
        **kwargs
    ) -> Any:
        """Create a chat completion using the Google Gemini API."""
        # Convert messages to Gemini format
        model_instance = genai.GenerativeModel(model or self.default_model)
        
        # Start a chat session
        chat = model_instance.start_chat()
        
        # System message handling - Add to first message instead of using generation_config
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        
        # Process functions if available
        if tools:
            function_declarations = self.format_tools(tools)
            model_instance.tools = function_declarations
        
        # Handle conversation history
        user_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # If there's a system message, prepend it to the first user message
        if system_message and user_messages:
            first_msg = user_messages[0]
            if first_msg["role"] == "user":
                # Create a system prompt prefix
                system_prefix = f"System Instructions: {system_message}\n\nUser Query: "
                user_messages[0]["content"] = system_prefix + first_msg["content"]
        
        # Send all messages in sequence
        for i, msg in enumerate(user_messages):
            # Skip empty messages
            if not msg.get("content"):
                continue
                
            if i == len(user_messages) - 1 and stream:
                # For the last message with streaming
                return chat.send_message(msg["content"], stream=True)
            else:
                # For all other messages
                chat.send_message(msg["content"])
        
        # For the last message without streaming
        return chat.last
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[FunctionDeclaration]:
        """Convert standard tool format to Google Gemini function declarations."""
        gemini_tools = []
        
        for tool in tools:
            if tool["type"] == "function":
                function_def = tool["function"]
                
                # Convert parameters to Google format
                parameters = {}
                if "parameters" in function_def and "properties" in function_def["parameters"]:
                    parameters = function_def["parameters"]
                
                # Create function declaration
                function_declaration = FunctionDeclaration(
                    name=function_def["name"],
                    description=function_def.get("description", ""),
                    parameters=parameters
                )
                
                gemini_tools.append(function_declaration)
        
        return gemini_tools
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from Google Gemini completion object."""
        # Check if it's a chat response
        content = ""
        if hasattr(completion, "text"):
            content = completion.text
        elif hasattr(completion, "parts"):
            content = "".join(part.text for part in completion.parts if hasattr(part, "text"))
        
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": self._extract_tool_calls_from_completion(completion)
        }
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from a Google Gemini stream chunk."""
        result = {}
        
        try:
            # For standard text chunks
            if hasattr(chunk, "text"):
                result["content"] = chunk.text
            elif hasattr(chunk, "parts"):
                result["content"] = "".join(part.text for part in chunk.parts if hasattr(part, "text"))
            
            # Extract function calls if present
            function_calls = self._extract_tool_calls_from_completion(chunk)
            if function_calls:
                result["tool_calls"] = []
                for i, call in enumerate(function_calls):
                    result["tool_calls"].append({
                        "index": i,
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]
                        }
                    })
        except Exception:
            pass
        
        return result
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Google Gemini completion."""
        return self._extract_tool_calls_from_completion(completion)
    
    def _extract_tool_calls_from_completion(self, completion: Any) -> List[Dict[str, Any]]:
        """Helper to extract tool calls from Google Gemini completion."""
        tool_calls = []
        
        try:
            if hasattr(completion, "function_call"):
                # Single function call
                function_call = completion.function_call
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": function_call.name,
                        "arguments": json.dumps(function_call.args)
                    }
                })
            elif hasattr(completion, "candidates") and completion.candidates:
                # Check candidates for function calls
                for i, candidate in enumerate(completion.candidates):
                    if hasattr(candidate, "function_call"):
                        function_call = candidate.function_call
                        tool_calls.append({
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": function_call.name,
                                "arguments": json.dumps(function_call.args)
                            }
                        })
        except (AttributeError, IndexError):
            pass
        
        return tool_calls
