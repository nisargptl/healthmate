from .llm_factory import LLMFactory
from typing import List
from langchain_core.messages import SystemMessage, AIMessage
from .llm_interface import LLMInterface

class LLMManager:
    def __init__(self):
        self.llm_instance = LLMFactory.create_llm_instance()
        self.tools = []
    def bind_tools(self, tools: List):
        """
        Binds tools to the LLM instance if supported. This method allows dynamic tool binding based on the context.
        """
        if isinstance(self.llm_instance, LLMInterface):
            self.llm_instance.bind_tools(tools)
            self.tools = tools

    def reset_tools(self):
        """
        Resets the tool bindings on the LLM instance by unbinding all tools.
        This method clears both the internal `self.tools` list and any bindings in the LLM.
        """
        if isinstance(self.llm_instance, LLMInterface):
            self.llm_instance.unbind_tools()
        self.tools = []

    def generate_response(self, messages: List[SystemMessage], bind_tools: bool = False) -> AIMessage:
        """
        Generates a response using the configured LLM instance. Optionally binds tools if specified.
        :param messages: List of system messages to send to the LLM.
        :param bind_tools: Whether to use tools during this invocation.
        :return: AIMessage object with the response.
        """
        if bind_tools and self.tools:
            self.llm_instance.bind_tools(self.tools)

        response = self.llm_instance.invoke(messages)

        if bind_tools:
            self.reset_tools()

        return response
