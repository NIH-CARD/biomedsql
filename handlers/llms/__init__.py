import os
from handlers.llms.azure_openai_llm import AzureOpenAILLM
from handlers.llms.azure_ai_llm import AzureAILLM
from handlers.llms.async_azure_openai_llm import AsyncAzureOpenAILLM
from handlers.llms.gemini_llm import GeminiLLM
from handlers.llms.anthropic_llm import AnthropicLLM

AZURE_CLIENT = AzureOpenAILLM.initialize_llm_client(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-12-01-preview"
)

AZURE_AI_CLIENT = AzureAILLM.initialize_llm_client(
    api_key=os.environ["AZURE_AI_API_KEY"],
    endpoint=os.environ["AZURE_AI_ENDPOINT"]
)

ASYNC_AZURE_CLIENT = AsyncAzureOpenAILLM.initialize_llm_client(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-12-01-preview"
)

GEMINI_CLIENT = GeminiLLM.initialize_llm_client(
    api_key=os.environ["GEMINI_API_KEY"]
)

ANTHROPIC_CLIENT = AnthropicLLM.initialize_llm_client(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)