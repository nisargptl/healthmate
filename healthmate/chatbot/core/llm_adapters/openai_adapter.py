from typing import List
from langchain_openai import ChatOpenAI
from .llm_interface import LLMInterface
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class OpenAIAdapter(LLMInterface):
    def __init__(self, api_key, model_name="gpt-4", temperature=0):
        self.api_key = api_key
        self.model_name = model_name
        self.client = ChatOpenAI(api_key=self.api_key, model=self.model_name)
        self.bound_tools = []

    def bind_tools(self, tools):
        self.client = self.client.bind_tools(tools)
        return self

    def unbind_tools(self):
        self.client = ChatOpenAI(api_key=self.api_key, model=self.model_name)
        self.bound_tools = []

    def invoke(self, messages: List) -> AIMessage:
        return self.client.invoke(messages)
