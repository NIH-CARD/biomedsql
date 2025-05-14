import httpx
import anthropic
from typing import Optional
from dataclasses import dataclass
from typeguard import typechecked
from handlers.llms.base_llm import BaseLLM

@typechecked
@dataclass(frozen=True)
class AnthropicLLM(BaseLLM):
    """
    AnthropicLLM is a class for handling chat completion calls to Anthropic endpoints.
    Extension of the abstract BaseLLM class.
    """

    llm_client: anthropic.Anthropic

    @staticmethod
    def initialize_llm_client(api_key: str) -> Optional[anthropic.Anthropic]:
        """
        Initialize the Anthropic client with the given API key.
        
        Parameters:
        - api_key (str): The API key for Anthropic.


        Returns:
        - Optional[anthropic.Anthropic]: The Anthropic module with the API key set, or None if initialization fails.
        """

        try:
            return anthropic.Anthropic(api_key=api_key, http_client=httpx.Client(timeout=60))
        except Exception as e:
            print(f'Error initializing Anthropic model: {e}')

    def query(self, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
        """
        Query the Anthropic API with the specified parameters.

        Parameters:
        - model_name (str): The name of the Anthropic model to use.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The sampling temperature.
        - query_text (str): The user's input text.

        Returns:
        - Tuple[str, List[Message]]: The response from the language model and the updated chat history.
        """

        try:
            chat_completion = self.llm_client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": query_text}
                ]
            )
            response = chat_completion.content[0].text
            return response
        
        except Exception as e:
            error_message = f"Error in {model_name} response: {e}"
            return error_message