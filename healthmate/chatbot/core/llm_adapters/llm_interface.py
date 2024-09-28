# llminterface.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.messages import AIMessage

class LLMInterface(ABC):
    """
    Abstract Base Class for LLM Interface.
    This interface ensures that any LLM implementation (e.g., OpenAI, Azure, Google) can be used interchangeably.
    """
    @abstractmethod
    def bind_tools(self, tools):
        """
        Binds tools to the LLM instance (if the LLM supports it).
        
        Args:
            tools (list): List of tools to bind.
        
        Returns:
            The LLM instance with tools bound.
        """
        pass

    @abstractmethod
    def unbind_tools(self):
        """
        Unbinds or clears all tools from the LLM instance.
        """
        pass

    @abstractmethod
    def invoke(self, messages: List) -> AIMessage:
        """
        Sends the messages to the LLM instance and returns the response.
        
        Args:
            messages (List[SystemMessage]): A list of messages to send to the LLM.
        
        Returns:
            AIMessage: The response from the LLM.
        """
        pass
