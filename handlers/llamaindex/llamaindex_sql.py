import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from vertexai import generative_models
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine

from handlers import CREDENTIALS
from handlers.llamaindex import TABLE_SCHEMAS
from handlers.llms import OPENAI_API_KEY, GEMINI_API_KEY

@dataclass(frozen=True)
class LlamaIndexSQL():
    sql_agent: Optional[SQLTableRetrieverQueryEngine]

    @staticmethod
    def initialize_sql_agent(project_id: str, database_name: str, llm_provider: str, model_name: str):
        try:
            bq_connection_uri = f"bigquery://{project_id}/{database_name}"
            engine = create_engine(bq_connection_uri, credentials_path='config/service_account.json')
            sql_database = SQLDatabase(engine)

            if llm_provider == 'gemini':
                safety_config = {
                    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                }

                llm = GoogleGenAI(
                    api_key=GEMINI_API_KEY,
                    model=model_name,
                    safety_settings=safety_config
                )
                embeddings = GoogleGenerativeAIEmbeddings(
                    google_api_key=GEMINI_API_KEY,
                    model="models/gemini-embedding-exp-03-07"
                )
            
            elif llm_provider == 'openai':
                llm = OpenAI(
                    api_key=OPENAI_API_KEY,
                    model=model_name
                )
                embeddings = OpenAIEmbeddings(
                    api_key=OPENAI_API_KEY,
                    model="text-embedding-ada-002"
                )
            else:
                raise ValueError(f'Invalid LLM provider passed: {llm_provider}')

            embed_model = LangchainEmbedding(embeddings)
            Settings.embed_model = embed_model
            Settings.llm = llm

            table_node_mapping = SQLTableNodeMapping(sql_database)
            obj_index = ObjectIndex.from_objects(TABLE_SCHEMAS, table_node_mapping, VectorStoreIndex)

            query_engine = SQLTableRetrieverQueryEngine(
                sql_database,
                obj_index.as_retriever(similarity_top_k=2),
                embed_model=embed_model,
                llm=llm,
                synthesize_response=True
            )

            return query_engine
        except Exception as e:
            print(f'Error initalizing LlamaIndex query engine: {e}')
            return None
        
    def run_agent(self, question):
        try:
            input_prompt = f'Question: {question} Always include the UUID column in your SELECT statements, except in cases of questions where the COUNT and ORDER BY functions are needed.'
            response = self.sql_agent.query(input_prompt)
            final_sql_query = response.metadata.get("sql_query", "")
            sql_query_results = response.metadata.get("result", "")
            final_answer = str(response)

            return final_sql_query, sql_query_results, final_answer
        except Exception as e:
        # Log error and set error message as answer
            final_sql_query = ""
            sql_query_results = ""
            final_answer = f"Error occurred: {str(e)}"
            print(f"Error processing question '{question}': {e}")
            return final_sql_query, sql_query_results, final_answer