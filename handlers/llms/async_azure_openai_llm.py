from dataclasses import dataclass
from typeguard import typechecked
from typing import Optional
from openai import AsyncAzureOpenAI
from handlers.llms.base_llm import BaseLLM

@typechecked
@dataclass(frozen=True)
class AsyncAzureOpenAILLM(BaseLLM):
    """
    AsyncAzureOpenAILLM is a class for handling asynchronous chat completion calls to Azure OpenAI endpoints.
    Extension of the abstract BaseLLM class.
    """

    llm_client: AsyncAzureOpenAI

    @staticmethod
    def initialize_llm_client(api_key: str, endpoint: str, api_version: str = '2024-02-01') -> Optional[AsyncAzureOpenAI]:
        """
        Initialize the AsyncAzureOpenAI client with the given API key.
        
        Parameters:
        - api_key (str): The API key for AsyncAzureOpenAI.


        Returns:
        - Optional[AsyncAzureOpenAI]: The AsyncAzureOpenAI module with the API key set, or None if initialization fails.
        """

        if not api_key:
            raise ValueError("API key is missing.")
        try:
            return AsyncAzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        except Exception as e:
            print(f"Error initializing AsyncAzureOpenAI client: {e}")
            return None

    async def query_async(self, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
        """
        Query the chat.completions AsyncAzureOpenAI API asynchronously with the specified parameters.

        Parameters:
        - model_name (str): The name of the AzureOpenAI model to use.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The sampling temperature.
        - query_text (str): The user's input text.

        Returns:
        - Tuple[str, List[Message]]: The response from the language model and the updated chat history.
        """

        if self.llm_client is None:
            raise ValueError("LLM Client is missing.")

        try:
            if 'gpt-o3-mini' in model_name:
                chat_completion = await self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": query_text}
                    ]
                )
            else:
                chat_completion = await self.llm_client.chat.completions.create(
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