# rabble/core.py
# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union, Dict, Any, Optional

# Local imports
from .util import function_to_json, debug_print, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)
from .adapters.base import ModelAdapter
from .adapters.factory import ModelAdapterFactory

__CTX_VARS_NAME__ = "context_variables"


class Rabble:
    def __init__(self, client=None, provider="openai", model=None):
        """
        Initialize a Rabble client.
        
        Args:
            client: Optional client instance for the provider
            provider: The LLM provider (openai, anthropic, deepseek)
            model: Default model to use
        """
        if isinstance(client, ModelAdapter):
            self.adapter = client
        else:
            self.adapter = ModelAdapterFactory.create_adapter(
                provider=provider,
                client=client,
                model=model
            )

    def get_chat_completion(
        self,
        agent: Agent,
        history: List[Dict[str, Any]],
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
        **kwargs  # Add **kwargs to accept additional parameters
    ):
        """Get a chat completion using the configured adapter."""
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        # Create an adapter for this agent if needed, or use the default one
        adapter = self.adapter
        if agent.provider != "openai":  # FIXME If agent uses a different provider than the default
            adapter = ModelAdapterFactory.create_adapter(
                provider=agent.provider,
                model=model_override or agent.model
            )

        # Create parameters for the adapter call
        adapter_params = {
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
            "model": model_override or agent.model,
        }
        
        # Only include parallel_tool_calls if there are tools and it's explicitly enabled
        if tools and agent.parallel_tool_calls:
            adapter_params["parallel_tool_calls"] = agent.parallel_tool_calls
        
        # Add any additional parameters
        adapter_params.update(kwargs)
        
        # Call the adapter's chat completion method with the proper parameters
        return adapter.chat_completion(**adapter_params)

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call["function"]["arguments"])
            debug_print(
                debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, debug)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "tool_name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
        **kwargs
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
                **kwargs
            )

            # Create or get appropriate adapter for this agent
            adapter = self.adapter
            if active_agent.provider != "openai":
                adapter = ModelAdapterFactory.create_adapter(
                    provider=active_agent.provider,
                    model=model_override or active_agent.model
                )

            yield {"delim": "start"}
            for chunk in completion:
                delta = adapter.extract_stream_chunk(chunk)
                if "role" in delta and delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message["tool_calls"], active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
        **kwargs  # Add **kwargs to pass additional parameters to the adapter
    ) -> Response:
        """
        Run the agent on the given messages.
    
        Args:
        agent: The agent to run
        messages: The conversation history
        context_variables: Additional context variables
        model_override: Override the agent's model
        stream: Whether to stream the response
        debug: Whether to print debug information
        max_turns: Maximum number of turns to take
        execute_tools: Whether to execute tools
        **kwargs: Additional parameters to pass to the adapter
        
        Returns:
            A Response object
        """
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
                **kwargs  # Pass kwargs to run_and_stream
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:
            # Create or get appropriate adapter for this agent
            adapter = self.adapter
            if active_agent.provider != "openai":
                adapter = ModelAdapterFactory.create_adapter(
                    provider=active_agent.provider,
                    model=model_override or active_agent.model
                )

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
                **kwargs  # Pass kwargs to get_chat_completion
            )
        
            # Extract response using the adapter
            message = adapter.extract_response(completion)
            message["sender"] = active_agent.name
            debug_print(debug, "Received completion:", message)
        
            history.append(message)

            tool_calls = message.get("tool_calls", [])
            if not tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
