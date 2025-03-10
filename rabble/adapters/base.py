# rabble/adapters/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional

class ModelAdapter(ABC):
    """Base class for model adapters to standardize interactions with different LLM providers."""
    
    @abstractmethod
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
        Create a chat completion with the configured model.
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            tool_choice: Optional specification for which tool to use
            stream: Whether to stream the response
            model: Model to use, overrides the default
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A standardized completion response or stream
        """
        pass
    
    @abstractmethod
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert standard tool format to provider-specific format."""
        pass
    
    @abstractmethod
    def extract_response(self, completion: Any) -> Dict[str, Any]:
        """Extract response from provider-specific completion object."""
        pass
    
    @abstractmethod
    def extract_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Extract information from a stream chunk and convert to standard format."""
        pass
    
    @abstractmethod
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from provider-specific completion."""
        pass
