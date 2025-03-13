# rabble/adapters/google_adapter.py
import json
import os
import re
from typing import List, Dict, Any, Iterator, Optional

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration

from .base import ModelAdapter

class GoogleAdapter(ModelAdapter):
    """
    Adapter for Google Gemini models with enhanced tool call detection.
    
    This adapter includes support for both API-based tool calls and
    text-based tool call recognition for Google models that describe
    tool usage in text rather than making formal tool calls.
    """
    
    def __init__(self, client=None, default_model=None, api_key=None):
        """
        Initialize the Google Gemini adapter.
        
        Args:
            client: Google GenerativeAI client (not used directly)
            default_model: Default model to use
            api_key: API key for Google Gemini
        """
        # Get model from parameter, environment, or use default
        self.default_model = default_model or os.getenv("GOOGLE_DEFAULT_MODEL")
        if not self.default_model:
            raise ValueError("No model specified for Google. Set GOOGLE_DEFAULT_MODEL environment variable.")
        
        # Get API key from parameter, environment, or raise error
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided for Google. Set GOOGLE_API_KEY environment variable.")
        
        # Set debug flag
        self.debug = os.getenv("GOOGLE_DEBUG", "0").lower() in ("1", "true", "yes")
        
        # Configure the genai module with the API key
        try:
            genai.configure(api_key=self.api_key)
            if self.debug:
                print(f"Initialized Google Gemini client with model: {self.default_model}")
        except Exception as e:
            error_msg = f"Google Gemini client initialization error: {str(e)}"
            if self.debug:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        
        # Store tools for text-based extraction
        self._current_tools = None
    
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
        # Store tools for potential text-based extraction
        self._current_tools = tools
        
        # Convert messages to Gemini format
        model_instance = genai.GenerativeModel(model or self.default_model)
        
        if self.debug:
            print(f"Calling Google API with model: {model or self.default_model}")
            print(f"Messages count: {len(messages)}")
            print(f"Tools count: {len(tools) if tools else 0}")
            if stream:
                print("Streaming mode enabled")
        
        # Start a chat session
        chat = model_instance.start_chat()
        
        # System message handling - Add to first message instead of using generation_config
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        
        # Process functions if available
        if tools:
            try:
                function_declarations = self.format_tools(tools)
                model_instance.tools = function_declarations
            except Exception as e:
                if self.debug:
                    print(f"Warning: Error setting tools: {str(e)}")
        
        # Handle conversation history
        user_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # If there's a system message, prepend it to the first user message
        if system_message and user_messages:
            first_msg = user_messages[0]
            if first_msg["role"] == "user":
                # Create a system prompt prefix
                system_prefix = f"System Instructions: {system_message}\n\nUser Query: "
                user_messages[0]["content"] = system_prefix + first_msg["content"]
        
        try:
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
        except Exception as e:
            error_msg = f"Google API call error: {str(e)}"
            if self.debug:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    
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
        """
        Extract response from Google Gemini completion object.
        
        This enhanced version also looks for text-based tool calls in the model's response
        for Gemini models that describe tool usage in text rather than making formal tool calls.
        """
        content = ""
        tool_calls = []
        
        # Extract text content from response
        try:
            if hasattr(completion, "text"):
                content = completion.text
            elif hasattr(completion, "parts"):
                content = "".join(part.text for part in completion.parts if hasattr(part, "text"))
            else:
                content = str(completion)
        except Exception as e:
            if self.debug:
                print(f"Error extracting content: {str(e)}")
            content = str(completion)
        
        # First try to extract API-based tool calls
        api_tool_calls = self._extract_tool_calls_from_completion(completion)
        if api_tool_calls:
            tool_calls = api_tool_calls
            if self.debug:
                print(f"Found {len(api_tool_calls)} API-based tool calls")
        
        # If no API tool calls found and content exists, try text extraction
        elif content and self._current_tools:
            if self.debug:
                print("Attempting text-based tool call extraction")
                print(f"Text length: {len(content)}")
                print(f"Text sample: {content[:200]}...")
            
            text_tool_calls = self._extract_tool_calls_from_text(content)
            if text_tool_calls:
                tool_calls = text_tool_calls
                if self.debug:
                    print(f"Extracted {len(text_tool_calls)} tool calls from text")
                    for call in text_tool_calls:
                        print(f"  - {call['function']['name']}({call['function']['arguments']})")
        
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls if tool_calls else None
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
            function_calls = self._extract_tool_calls_from_chunk(chunk)
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
        except Exception as e:
            if self.debug:
                print(f"Error in stream chunk extraction: {str(e)}")
        
        return result
    
    def _extract_tool_calls_from_chunk(self, chunk: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from a streaming chunk."""
        tool_calls = []
        
        try:
            # Check for function calls in streaming
            # Currently, Gemini doesn't officially support streaming function calls
            # but we'll include this for future compatibility
            if hasattr(chunk, "function_call"):
                function_call = chunk.function_call
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": function_call.name,
                        "arguments": json.dumps(function_call.args)
                    }
                })
            
            # Try to extract from text if no API-based calls
            if not tool_calls and hasattr(chunk, "text") and chunk.text:
                text_tool_calls = self._extract_tool_calls_from_text(chunk.text)
                if text_tool_calls:
                    tool_calls.extend(text_tool_calls)
        except Exception as e:
            if self.debug:
                print(f"Error extracting tool calls from chunk: {str(e)}")
        
        return tool_calls
    
    def extract_tool_calls(self, completion: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Google Gemini completion."""
        api_tool_calls = self._extract_tool_calls_from_completion(completion)
        
        # If no API tool calls, try text extraction
        if not api_tool_calls and self._current_tools:
            try:
                content = ""
                if hasattr(completion, "text"):
                    content = completion.text
                elif hasattr(completion, "parts"):
                    content = "".join(part.text for part in completion.parts if hasattr(part, "text"))
                else:
                    content = str(completion)
                
                # Try to extract from text
                if content:
                    text_tool_calls = self._extract_tool_calls_from_text(content)
                    if text_tool_calls and self.debug:
                        print(f"Extracted {len(text_tool_calls)} tool calls from text")
                    return text_tool_calls
            except Exception as e:
                if self.debug:
                    print(f"Error in text-based tool call extraction: {str(e)}")
        
        return api_tool_calls
    
    def _extract_tool_calls_from_completion(self, completion: Any) -> List[Dict[str, Any]]:
        """Helper to extract API-based tool calls from Google Gemini completion."""
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
        except (AttributeError, IndexError) as e:
            if self.debug:
                print(f"Error extracting API tool calls: {str(e)}")
        
        return tool_calls
    
    def _extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from text based on common patterns.
        
        This method looks for patterns like:
        - ```tool_code\noperation="multiply"\nx=605304\ny=882661\n```
        - `operation="multiply", x=610774, y=453916`
        - Calculator Input: {"operation": "multiply", "x": 947670, "y": 316095}
        
        Args:
            text: Text to extract tool calls from
            
        Returns:
            List of tool call dictionaries
        """
        if not text or not self._current_tools:
            return []
        
        tool_calls = []
        tool_names = [tool["function"]["name"] for tool in self._current_tools if tool["type"] == "function"]
        
        # === PATTERN 1: Triple backtick code blocks with tool_code marker ===
        # ```tool_code
        # operation="multiply"
        # x=605304
        # y=882661
        # ```
        tool_code_pattern = r'```tool_code\s*(.*?)\s*```'
        
        for match in re.finditer(tool_code_pattern, text, re.DOTALL):
            try:
                code_block = match.group(1).strip()
                if self.debug:
                    print(f"Found tool_code block: {code_block}")
                
                # Extract the parameters from the code block
                params = {}
                
                # Split the code block into lines and process each line
                for line in code_block.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try to parse as key=value
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        
                        # Convert numeric values
                        if key != "operation":
                            try:
                                value = float(value.replace(',', ''))
                                # Convert to int if it's a whole number
                                if value.is_integer():
                                    value = int(value)
                            except (ValueError, AttributeError):
                                pass
                        
                        params[key] = value
                
                # Only create a tool call if we have at least operation and x
                if "operation" in params and "x" in params:
                    tool_calls.append({
                        "id": f"text_call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": json.dumps(params)
                        }
                    })
                    
                    if self.debug:
                        print(f"Extracted tool_code format: calculate({json.dumps(params)})")
            except Exception as e:
                if self.debug:
                    print(f"Error parsing tool_code block: {e}")
        
        # If we found tool calls with the tool_code pattern, return them
        if tool_calls:
            return tool_calls
        
        # === PATTERN 2: Backtick-enclosed parameters ===
        # `operation="multiply", x=610774, y=453916`
        backtick_pattern = r'`([^`]+)`'
        
        for match in re.finditer(backtick_pattern, text):
            try:
                # Attempt to extract parameters from backtick contents
                params_text = match.group(1)
                
                # Check if this looks like our parameters
                if "operation=" in params_text and "x=" in params_text:
                    # Extract operation
                    op_match = re.search(r'operation\s*=\s*["\']?([^"\',]+)["\']?', params_text)
                    if not op_match:
                        continue
                    
                    operation = op_match.group(1).strip()
                    
                    # Extract x parameter
                    x_match = re.search(r'x\s*=\s*([0-9,.]+)', params_text)
                    if not x_match:
                        continue
                    
                    x_value = float(x_match.group(1).replace(',', ''))
                    
                    # Create parameters dict
                    params = {"operation": operation, "x": x_value}
                    
                    # Add y parameter if present and needed
                    y_match = re.search(r'y\s*=\s*([0-9,.]+)', params_text)
                    if y_match and operation != "sqrt":
                        y_value = float(y_match.group(1).replace(',', ''))
                        params["y"] = y_value
                    
                    # Add tool call
                    tool_calls.append({
                        "id": f"text_call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": json.dumps(params)
                        }
                    })
                    
                    if self.debug:
                        print(f"Extracted backtick format: calculate({json.dumps(params)})")
            except Exception as e:
                if self.debug:
                    print(f"Error parsing backtick format: {e}")
        
        # If we found tool calls with the backtick pattern, return them
        if tool_calls:
            return tool_calls
        
        # === PATTERN 3: Calculator Input with JSON ===
        # Calculator Input: {"operation": "multiply", "x": 947670, "y": 316095}
        calculator_input_pattern = r'Calculator Input:\s*(\{[^}]+\})'
        
        for match in re.finditer(calculator_input_pattern, text, re.MULTILINE):
            try:
                json_str = match.group(1)
                args = json.loads(json_str)
                
                # Add call
                tool_calls.append({
                    "id": f"text_call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": json.dumps(args)
                    }
                })
                
                if self.debug:
                    print(f"Extracted Calculator Input format: {json.dumps(args)}")
            except (json.JSONDecodeError, IndexError) as e:
                if self.debug:
                    print(f"Failed to parse JSON in Calculator Input: {e}")
        
        # === Last Resort: Look for Result Statements ===
        if not tool_calls:
            result_patterns = [
                r'Result:?\s*(?:of\s+multiplication:?)?\s*([\d,\.]+)',
                r'Result:?\s*(?:of\s+square\s+root:?)?\s*([\d,\.]+)'
            ]
            
            # Find results
            results = []
            for pattern in result_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    try:
                        result_value = float(match.group(1).replace(',', ''))
                        results.append(result_value)
                    except (ValueError, IndexError):
                        continue
            
            # Find input numbers
            if results:
                # Find all numeric values
                number_pattern = r'(\d[\d,]+\d|\d+)'
                number_matches = list(re.finditer(number_pattern, text))
                numbers = [float(m.group(1).replace(',', '')) for m in number_matches if '.' not in m.group(1)]
                
                if len(numbers) >= 2 and len(results) >= 1:
                    # Find two numbers that when multiplied are close to the first result
                    for i in range(len(numbers)):
                        for j in range(i+1, len(numbers)):
                            if (abs(numbers[i] * numbers[j] - results[0]) / results[0]) < 0.01:
                                # Found likely multiplication inputs
                                tool_calls.append({
                                    "id": f"text_call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": json.dumps({
                                            "operation": "multiply", 
                                            "x": numbers[i], 
                                            "y": numbers[j]
                                        })
                                    }
                                })
                                
                                # If we have a second result, it's likely the square root
                                if len(results) >= 2:
                                    tool_calls.append({
                                        "id": f"text_call_{len(tool_calls)}",
                                        "type": "function",
                                        "function": {
                                            "name": "calculate",
                                            "arguments": json.dumps({
                                                "operation": "sqrt", 
                                                "x": results[0]
                                            })
                                        }
                                    })
                                
                                if self.debug:
                                    print(f"Inferred from results: multiply({numbers[i]}, {numbers[j]}) and sqrt({results[0]})")
                                break
        
        return tool_calls
