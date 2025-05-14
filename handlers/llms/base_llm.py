from abc import ABC, abstractmethod
from dataclasses import dataclass
from typeguard import typechecked

@typechecked
@dataclass(frozen=True)
class BaseLLM(ABC):
    """
    BaseLLM is an abstract class for language models.
    Provides a generic interface for generating responses.
    """

    @abstractmethod
    def query(self, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
        """
        Query the language model.

        Parameters:
        - model_name (str): The name of the model.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The sampling temperature.
        - query_text (str): The user's input text.

        Returns:
        - str: The response from the language model.
        """
        
        pass