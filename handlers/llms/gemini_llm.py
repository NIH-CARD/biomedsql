from dataclasses import dataclass
from typeguard import typechecked
from typing import Optional
from google import genai
from handlers.llms.base_llm import BaseLLM

@typechecked
@dataclass(frozen=True)
class GeminiLLM(BaseLLM):
    """
    GeminiLLM is a class for handling synchronous and asynchronous generate content calls to Gemini endpoints.
    Extension of the abstract BaseLLM class.
    """

    llm_client: genai.Client

    @staticmethod
    def initialize_llm_client(api_key: str) -> Optional[genai.Client]:
        """Initialize the Gemini client."""
        if not api_key:
            raise ValueError("API key is missing.")
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            return None

    def query(self, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
        """
        Query the Gemini API synchronously.
        
        Parameters:
        - model_name (str): The name of the Gemini model to use.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The sampling temperature.
        - query_text (str): The user's input text.

        Returns:
        - str: The response from the language model and the updated chat history.
        """

        if self.llm_client is None:
            raise ValueError("LLM Client is missing.")

        try:
            config = genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                safety_settings = [genai.types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                    ),genai.types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                    ),genai.types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                    ),genai.types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )]
            )

            response = self.llm_client.models.generate_content(
                model=model_name,
                contents=query_text,
                config=config
            )
                            
            response_text = response.text
            return response_text
            
        except Exception as e:
            error_message = f"Error in Gemini response: {e}"
            return error_message
        
    async def query_async(self, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
        """
        Query the Gemini API synchronously.
        
        Parameters:
        - model_name (str): The name of the Gemini model to use.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The sampling temperature.
        - query_text (str): The user's input text.

        Returns:
        - str: The response from the language model and the updated chat history.
        """

        if self.llm_client is None:
            raise ValueError("LLM Client is missing.")

        try:
            config = genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                safety_settings = [genai.types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                    ),genai.types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                    ),genai.types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                    ),genai.types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )]
            )
            
            response = await self.llm_client.aio.models.generate_content(
                model=model_name,
                contents=query_text,
                config=config
            )
            response_text = response.text
            return response_text
            
        except Exception as e:
            error_message = f"Error in Gemini response: {e}"
            return error_message