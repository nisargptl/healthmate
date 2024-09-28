import os
from .openai_adapter import OpenAIAdapter
from .googleai_adapter import GoogleAIAdapter
from .mistralai_adapter import MistralAIAdapter
from dotenv import load_dotenv
from .llm_interface import LLMInterface
load_dotenv()

class LLMFactory:
    @staticmethod
    def create_llm_instance() -> LLMInterface:
        """
        Creates an LLM instance based on environment variables.
        """
        llm_model = os.getenv('LLM_MODEL')  
        api_key = os.getenv('API_KEY')
        model_name = os.getenv('MODEL_NAME')
        temperature = float(os.getenv('TEMPERATURE', 0))
        if llm_model == 'openai':
            return OpenAIAdapter(api_key=api_key, model_name=model_name, temperature=temperature)
        elif llm_model == 'googleai':
            return GoogleAIAdapter(api_key=api_key, model_name=model_name, temperature=temperature)
        elif llm_model == 'mistralai':
            return MistralAIAdapter(api_key=api_key, model_name=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_model}")
