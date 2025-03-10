# rabble/adapters/openai_adapter.py
import json
from typing import List, Dict, Any, Iterator, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from .base import ModelAdapter

class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models."""
    
    def __init__(self, client=None, default_model="gpt-4o"):
        """
        Initialize the OpenAI adapter.
        
        Args:
            client: OpenAI client instance
            default_model: Default model to use
        """
        self.client = client or OpenAI()
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
        """Create a chat completion using the OpenAI API."""
        create_params = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        if tools:
            create_params["tools"] = self.format_tools(tools)
            if tool_choice:
                create_params["tool_choice"] = tool_choice
            if "parallel_tool_calls" in kwargs:
                create_params["parallel_tool_calls"] = kwargs["parallel_tool_calls"]
        
        return self.client.chat.completions.create(**create_params)
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OpenAI tools are already in the expected format."""
        return tools or None
    
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from OpenAI completion object."""
        message = completion.choices[0].message
        return json.loads(message.model_dump_json())
    
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from an OpenAI stream chunk."""
        return json.loads(chunk.choices[0].delta.json())
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from OpenAI completion."""
        message = completion.choices[0].message
        if not message.tool_calls:
            return []
        
        tool_calls = []
        for tool_call in message.tool_calls:
            tool_calls.append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
        
        return tool_calls
