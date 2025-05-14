from typing import Optional
from dataclasses import dataclass
from typeguard import typechecked

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

from handlers.llms.base_llm import BaseLLM

@typechecked
@dataclass(frozen=True)
class AzureAILLM(BaseLLM):
    """
    AsyncAILLM is a class for handling completion calls to Azure AI endpoints.
    Extension of the abstract BaseLLM class.
    """

    llm_client: ChatCompletionsClient

    @staticmethod
    def initialize_llm_client(api_key: str, endpoint: str) -> Optional[ChatCompletionsClient]:
        """
        Initialize the AzureAI client with the given API and endpoint key.
        
        Parameters:
        - api_key (str): The API key for AzureAI.
        - endpoint (str): The endpoint for AzureAI.
        - api_version (str): The OpenAI API version to use for AzureAI

        Returns:
        - Optional[AzureAI]: The AzureAI module with the API key, API version, and endpoint set, or None if initialization fails.
        """

        if not api_key:
            raise ValueError("API key is missing.")
        if not endpoint:
            raise ValueError("Endpoint is missing")
        try:
            return ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        except Exception as e:
            print(f"Error initializing AzureAI client: {e}")
            return None
    
    def query(self, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
        """
        Query the complete AzureAI API with the specified parameters.

        Parameters:
        - model_name (str): The name of the AzureAI model to use.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The sampling temperature.
        - query_text (str): The user's input text.

        Returns:
        - str: The response from the language model.
        """

        if self.llm_client is None:
            raise ValueError("LLM Client is missing.")

        try:
            chat_completion = self.llm_client.complete(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[UserMessage(content=query_text)]
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            error_message = f"Error in {model_name} response: {e}"
            return error_message