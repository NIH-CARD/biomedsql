import os
import json
import re
import tiktoken
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
from typeguard import typechecked
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from handlers.llms import (
    AZURE_CLIENT,
    AZURE_AI_CLIENT,
    ASYNC_AZURE_CLIENT,
    GEMINI_CLIENT,
    ANTHROPIC_CLIENT
)
from handlers.llms.azure_ai_llm import AzureAILLM
from handlers.llms.azure_openai_llm import AzureOpenAILLM
from handlers.llms.gemini_llm import GeminiLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ID = "card-ai-389220"  # Biomedical dataset project
DATASET_NAME = "bio_sql_benchmark"  # Biomedical benchmark dataset
QUERY_LIMIT = 10  # Limit for query results
MAX_RETRIES = 5

# ------ BigQuery Handler Class ------

def count_tokens_tiktoken(string: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    string = str(string) if string is not None else ""
    num_tokens = len(encoding.encode(string))
    return num_tokens

@typechecked
@dataclass(frozen=True)
class BigQuery:
    """
    BigQuery handler for database operations.
    """
    bigquery_client: bigquery.Client

    @staticmethod
    def initialize_bigquery_client(
        credentials_path: Optional[str] = 'config/service_account.json',
        project: Optional[str] = PROJECT_ID
    ) -> Optional[Any]:
        """Initialize BigQuery client with credentials."""
        try:
            print(f"\n{'='*30} INITIALIZING BIGQUERY CLIENT {'='*30}")
            print(f"Project ID: {project}")
            print(f"Credentials path: {credentials_path}")
            
            if os.path.exists(credentials_path):
                print(f"Credentials file found at {credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                client = bigquery.Client(project=project)
                print("Successfully initialized BigQuery client with credentials file")
                return client
            else:
                print(f"Credentials file not found at {credentials_path}, using default credentials")
                logger.warning(f"Credentials file not found at {credentials_path}, using default credentials")
                try:
                    client = bigquery.Client(project=project)
                    print("Successfully initialized BigQuery client with default credentials")
                    return client
                except Exception as e:
                    print(f"Failed to initialize with default credentials: {e}")
                    logger.error(f"Failed to initialize with default credentials: {e}")
                    return 
        except Exception as e:
            error_msg = f'Error initializing BigQuery client: {e}'
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            raise
            
    def query_db(self, query: str) -> Tuple[bool, Any]:
        """
        Execute a SQL query against BigQuery.
        
        Returns:
            Tuple[bool, Any]: (success, results or error message)
        """
        print(f"\n{'='*30} EXECUTING QUERY {'='*30}")
        print(f"Query to execute: {query}")
        
        try:
            results = self.bigquery_client.query(query).result()
            df = results.to_dataframe()
            
            print(f"Query executed successfully")
            print(f"Result shape: {df.shape}")
            if not df.empty:
                print(f"Column names: {list(df.columns)}")
                print(f"Sample result:\n{df.head(3)}")
            else:
                print("Result is empty")
                
            return True, df
        except Exception as e:
            error_message = str(e)
            print(f"Query execution failed: {error_message}")
            logger.error(f'Failed to run query: {error_message}')
            return False, error_message
    
    def verify_query(self, query: str) -> Tuple[bool, str]:
        """
        Verify if a SQL query is valid without executing it.
        Uses BigQuery's dry run feature.
        
        Returns:
            Tuple[bool, str]: (is_valid, "" or error message)
        """
        print(f"\n{'='*30} VERIFYING QUERY {'='*30}")
        print(f"Query to verify: {query}")
        
        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            self.bigquery_client.query(query, job_config=job_config)
            print("Query syntax verification successful")
            return True, ""
        except Exception as e:
            error_message = str(e)
            print(f"Query syntax verification failed: {error_message}")
            
            if "Not found: Table" in error_message:
                table_name = re.search(r"Not found: Table ([^\s]+)", error_message)
                if table_name:
                    error_message += f"\nThe table '{table_name.group(1)}' does not exist in the dataset."
            
            return False, error_message
    
    def get_schema_info(self, dataset_id: str = DATASET_NAME) -> Dict[str, List[Dict[str, str]]]:
        """
        Get database schema information including tables and their columns.
        
        Returns:
            Dict mapping table names to their column details
        """
        schema_info = {}
        
        try:
            print(f"\n{'='*30} FETCHING SCHEMA INFORMATION {'='*30}")
            print(f"Fetching schema for dataset: {dataset_id}")
            
            dataset_ref = self.bigquery_client.dataset(dataset_id)
            tables = list(self.bigquery_client.list_tables(dataset_ref))
            
            print(f"Found {len(tables)} tables in dataset")
            
            if len(tables) == 0:
                print("WARNING: No tables found in dataset. Schema will be empty.")
                return {}
            
            for table in tables:
                print(f"Processing table: {table.table_id}")
                table_ref = self.bigquery_client.get_table(table)
                columns = []
                
                for field in table_ref.schema:
                    columns.append({
                        "name": field.name,
                        "type": field.field_type,
                        "description": field.description
                    })
                
                schema_info[table.table_id] = columns
                print(f"Added {len(columns)} columns for table {table.table_id}")
            
            print(f"Final schema has {len(schema_info)} tables")
            print(f"Schema information: {json.dumps(schema_info, indent=2)}")
            print(f"{'='*80}\n")
            return schema_info
            
        except Exception as e:
            error_msg = f"Error fetching schema: {e}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            raise

# ------ ReACT Agent Models ------

class Action(str, Enum):
    """Available actions for the ReACT agent."""
    VERIFY_SQL = "verify_sql"
    EXECUTE_SQL = "execute_sql"
    FINAL_ANSWER = "final_answer"

class ReasoningStep(BaseModel):
    """Model for each reasoning step in the ReACT process."""
    thought: str = Field(..., description="Reasoning about the current state")
    action: Action = Field(..., description="Next action to take")
    action_input: str = Field(..., description="Input for the selected action")

class SQLAgentResponse(BaseModel):
    """Model for the final response from the SQL agent."""
    question: str = Field(..., description="Original question asked")
    sql_query: str = Field(..., description="SQL query that answers the question")
    explanation: str = Field(..., description="Explanation of how the SQL query answers the question")
    results: Any = Field(None, description="Results of executing the SQL query")
    execution_time: float = Field(0.0, description="Time taken to execute the query")
    tokens: int = Field(0, description="Number of input tokens used")

# ------ LLM Interface Class ------

class LLMInterface:
    """Interface for the LLM model used in the ReACT agent."""
    
    def __init__(self, model_type: str = "openai", model_name: str = "gpt-4"):
        """Initialize the LLM interface."""
        self.model_type = model_type
        self.model_name = model_name
        
        # Initialize the appropriate LLM client based on model_type
        if model_type == "azure_openai":
            self.client = AzureOpenAILLM(llm_client = AZURE_CLIENT)
        elif model_type == "gemini":
            self.client = GeminiLLM(llm_client = GEMINI_CLIENT)
        elif model_type == "anthropic":
            self.client = ANTHROPIC_CLIENT
        elif model_type == "azure_ai":
            self.client = AzureAILLM(llm_client = AZURE_AI_CLIENT)
        elif model_type == "async_azure":
            self.client = ASYNC_AZURE_CLIENT
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def generate_next_step(self, 
                         question: str, 
                         schema_info: Dict, 
                         history: List[Dict],
                         system_prompt: str,
                         dataset_name: str = DATASET_NAME) -> str:
        """
        Generate the next reasoning step for the ReACT agent.
        
        Args:
            question: The user's original question
            schema_info: Database schema information
            history: Previous reasoning steps and observations
            system_prompt: The system prompt to guide the LLM
            dataset_name: The dataset name to use in queries
            
        Returns:
            String containing the next reasoning step in JSON format
        """
        # Format the schema info for the prompt
        schema_str = json.dumps(schema_info, indent=2)
        
        # Format the conversation history
        history_str = ""
        for item in history:
            if "thought" in item:
                history_str += f"Thought: {item['thought']}\n"
                history_str += f"Action: {item['action']}\n"
                history_str += f"Action Input: {item['action_input']}\n"
            if "observation" in item:
                history_str += f"Observation: {item['observation']}\n\n"
        
        print(f"\n{'='*30} GENERATING NEXT STEP {'='*30}")
        print(f"Question: {question}")
        print(f"History length: {len(history)} items")
        print(f"Schema has {len(schema_info)} tables")
        print(f"Using dataset name: {dataset_name}")
        
        prompt = f"""
        {system_prompt}
        
        Question: {question}
        
        Database Schema:
        ```json
        {schema_str}
        ```
        
        Dataset Name: {dataset_name}
        
        Reasoning History:
        {history_str}
        
        Continue the reasoning process with the next step:
        """
        
        try:
            tokens = count_tokens_tiktoken(string=prompt)

            response = self.client.query(
                model_name=self.model_name,
                max_tokens=1500,
                temperature=0.2,
                query_text=prompt
            )
            return response, tokens
        except Exception as e:
            error_msg = f"ERROR calling LLM API: {e}"
            print(error_msg)
            logger.error(error_msg)
            raise
    
    def generate_final_response(self, 
                             question: str, 
                             sql_query: str, 
                             results: Any,
                             execution_time: float) -> str:
        """
        Generate a human-readable explanation for the final answer.
        
        Args:
            question: The user's original question
            sql_query: The successful SQL query
            results: The query results
            execution_time: Time taken to execute the query
            
        Returns:
            A human-readable explanation of the results
        """
        results_str = str(results.head(10)) if isinstance(results, pd.DataFrame) else str(results)
        
        prompt = f"""
        You are a helpful data analyst explaining SQL query results.
        
        Question: {question}
        
        SQL Query:
        ```sql
        {sql_query}
        ```
        
        Query Results (first 10 rows):
        ```
        {results_str}
        ```
        
        Query execution time: {execution_time:.2f} seconds
        
        Please provide a clear and concise explanation of how this SQL query answers the original question,
        and summarize the key findings from the results.
        """
        
        try:
            tokens = count_tokens_tiktoken(string=prompt)

            response = self.client.query(
                model_name=self.model_name,
                max_tokens=1000,
                temperature=0.3,
                query_text=prompt
            )
            return response, tokens
        except Exception as e:
            error_msg = f"ERROR generating final response: {e}"
            print(error_msg)
            logger.error(error_msg)
            raise 