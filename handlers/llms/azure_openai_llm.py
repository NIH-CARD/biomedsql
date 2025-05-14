from dataclasses import dataclass
from typeguard import typechecked
from typing import Optional
from openai import AzureOpenAI
from handlers.llms.base_llm import BaseLLM

@typechecked
@dataclass(frozen=True)
class AzureOpenAILLM(BaseLLM):
    """
    AzureOpenAILLM is a class for handling synchronous chat completion calls to Azure OpenAI endpoints.
    Extension of the abstract BaseLLM class.
    """

    llm_client: AzureOpenAI

    @staticmethod
    def initialize_llm_client(api_key: str, endpoint: str, api_version: str = '2024-02-01') -> Optional[AzureOpenAI]:
        """
        Initialize the AzureOpenAI client with the given API and endpoint key.
        
        Parameters:
        - api_key (str): The API key for AzureOpenAI.
        - endpoint (str): The endpoint for AzureOpenAI.
        - api_version (str): The OpenAI API version to use for AzureOpenAI

        Returns:
        - Optional[AzureOpenAI]: The AzureOpenAI module with the API key, API version, and endpoint set, or None if initialization fails.
        """

        if not api_key:
            raise ValueError("API key is missing.")
        if not endpoint:
            raise ValueError("Endpoint is missing")
        try:
            return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        except Exception as e:
            print(f"Error initializing AzureOpenAI client: {e}")
            return None

    def query(self, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
        """
        Query the chat.completions AzureOpenAI API with the specified parameters.

        Parameters:
        - model_name (str): The name of the AzureOpenAI model to use.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The sampling temperature.
        - query_text (str): The user's input text.

        Returns:
        - str: The response from the language model.
        """

        if self.llm_client is None:
            raise ValueError("LLM Client is missing.")

        try:
            if 'gpt-o3-mini' in model_name:
                chat_completion = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": query_text}
                    ]
                )
            else:
                chat_completion = self.llm_client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": query_text}
                    ]
                )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            error_message = f"Error in {model_name} response: {e}"
            return error_message