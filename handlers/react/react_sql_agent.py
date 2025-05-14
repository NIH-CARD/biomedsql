import time
import json
import re
import pandas as pd
from dataclasses import dataclass
from .react_handlers import (
    BigQuery,
    LLMInterface,
    SQLAgentResponse,
    Action,
    DATASET_NAME,
    PROJECT_ID,
    MAX_RETRIES
)

@dataclass
class ReACTSQLAgent:
    """
    ReACT-based SQL agent for BigQuery.
    
    Uses a reasoning loop to iteratively refine SQL queries until a correct answer is found.
    """
    bq_handler: BigQuery
    llm_interface: LLMInterface
    
    @staticmethod
    def initialize_agent(
        model_type: str = "gemini",
        model_name: str = "gemini-2.0-flash",
        credentials_path: str = "config/service_account.json",
        project_id: str = PROJECT_ID
    ):
        """Factory method to create and initialize a ReACTSQLAgent."""
        try:
            bq_client = BigQuery.initialize_bigquery_client(
                credentials_path=credentials_path,
                project=project_id
            )
            
            if not bq_client:
                raise ValueError("Failed to initialize BigQuery client")
            
            bq_handler = BigQuery(bigquery_client=bq_client)
            llm_interface = LLMInterface(model_type=model_type, model_name=model_name)
            
            return ReACTSQLAgent(bq_handler=bq_handler, llm_interface=llm_interface)
        except Exception as e:
            error_msg = f"Failed to initialize ReACTSQLAgent: {e}"
            raise ValueError(error_msg)
    
    def run(self, question: str) -> SQLAgentResponse:
        """
        Run the ReACT SQL agent to answer a question.
        
        Args:
            question: Natural language question about the database
            
        Returns:
            SQLAgentResponse with query, explanation and results
        """
        # Start timing the entire process
        start_time = time.time()
        
        # Get schema information
        try:
            schema_info = self.bq_handler.get_schema_info(DATASET_NAME)
        except Exception as e:
            return SQLAgentResponse(
                question=question,
                sql_query="-- Error getting schema information",
                explanation=f"Failed to retrieve database schema: {str(e)}",
                results=None,
                execution_time=0.0
            )
        
        # ReACT system prompt
        system_prompt = f"""
        You are an expert SQL agent that uses step-by-step reasoning to answer questions about data in a BigQuery database.
        
        IMPORTANT: The dataset name is "{DATASET_NAME}". Always qualify table names with this dataset name.
        Example: SELECT * FROM {DATASET_NAME}.table_name
        
        Follow these steps:
        1. Think about how to translate the question into a SQL query
        2. Decide which tables and columns are needed
        3. Write a SQL query with explanatory comments
        4. Verify the query syntax before executing
        5. If the query has errors, fix them and try again
        6. Once the query is successful, explain the results clearly
        7. Always include the UUID column in your SELECT statements, except in cases of questions where the COUNT and ORDER BY functions are needed.
        8. Unless the user explicitly requests a different LIMIT, default your queries to LIMIT 100.
        9. Avoid SELECT *; select only the necessary columns to answer the user’s query.
        10. Ensure that any disease names that contain an apostrophe in the query are surrounded by double quotes (e.g., "Alzheimer's Disease").
        
        Your output MUST be a JSON object with these fields:
        {{
          "thought": "Your reasoning about how to answer the question",
          "action": "One of 'verify_sql', 'execute_sql', or 'final_answer'",
          "action_input": "For verify_sql/execute_sql: the SQL query; For final_answer: explanation of the results"
        }}
        
        IMPORTANT: 
        - Your response must include valid JSON that I can parse. 
        - Do not include any explanations outside the JSON object. 
        - Always qualify table names with the dataset name "{DATASET_NAME}."
        
        Make sure your SQL queries follow BigQuery SQL syntax and include helpful inline comments.
        """
        
        # Initialize the reasoning loop
        reasoning_history = []
        final_sql_query = None
        final_results = None
        total_tokens = 0
        execution_time = 0.0
        
        # Track all SQL queries attempted
        all_sql_queries = []
        
        # ReACT loop - maximum of MAX_RETRIES iterations
        for iteration in range(MAX_RETRIES):
            try:
                # Get next reasoning step from LLM
                step_text, tokens = self.llm_interface.generate_next_step(
                    question=question,
                    schema_info=schema_info,
                    history=reasoning_history,
                    system_prompt=system_prompt,
                    dataset_name=DATASET_NAME
                )

                total_tokens += tokens
                
                # Parse the response
                step_data = None
                try:
                    step_data = json.loads(step_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', step_text, re.DOTALL)
                    if json_match:
                        try:
                            step_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                
                if not step_data:
                    raise ValueError("Failed to parse JSON response from LLM")
                
                # Extract fields
                thought = step_data.get("thought", "")
                action = step_data.get("action", "")
                action_input = step_data.get("action_input", "")
                
                # Validate action
                if action not in [a.value for a in Action]:
                    raise ValueError(f"Invalid action: {action}")
                
                # Add to history
                reasoning_history.append({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input
                })
                
                # Process the action
                observation = None
                
                if action == Action.VERIFY_SQL or action == Action.EXECUTE_SQL:
                    all_sql_queries.append({
                        "iteration": iteration + 1,
                        "action": action,
                        "query": action_input
                    })
                
                if action == Action.VERIFY_SQL:
                    is_valid, error_msg = self.bq_handler.verify_query(action_input)
                    if is_valid:
                        observation = "The query syntax is valid."
                    else:
                        observation = f"The query has syntax errors: {error_msg}"
                
                elif action == Action.EXECUTE_SQL:
                    query_start_time = time.time()
                    success, results = self.bq_handler.query_db(action_input)
                    query_end_time = time.time()
                    execution_time = query_end_time - query_start_time
                    
                    if success:
                        final_sql_query = action_input
                        final_results = results
                        
                        if isinstance(results, pd.DataFrame):
                            result_preview = results.head(5).to_string()
                            total_rows = len(results)
                            observation = f"Query executed successfully. Returned {total_rows} rows. Preview:\n{result_preview}"
                        else:
                            observation = f"Query executed successfully. Results: {results}"
                    else:
                        observation = f"Query execution failed: {results}"
                
                elif action == Action.FINAL_ANSWER:
                    if not final_sql_query or final_results is None:
                        observation = "Cannot provide final answer without successful query execution."
                    else:
                        break
                
                # Add observation to history
                reasoning_history.append({"observation": observation})
                
            except Exception as e:
                error_msg = f"Error in reasoning step {iteration+1}: {str(e)}"
                reasoning_history.append({
                    "observation": f"Error processing the last step: {str(e)}"
                })
        
        # Generate final explanation
        total_time = time.time() - start_time
        
        if final_sql_query and final_results is not None:
            try:
                explanation, tokens = self.llm_interface.generate_final_response(
                    question=question,
                    sql_query=final_sql_query,
                    results=final_results,
                    execution_time=execution_time
                )
                total_tokens += tokens
            except Exception as e:
                explanation = f"Failed to generate explanation: {str(e)}"
        else:
            explanation = "Failed to find a working SQL query to answer this question."
            final_sql_query = "-- No working query found"
            final_results = pd.DataFrame()
        
        return SQLAgentResponse(
            question=question, 
            sql_query=final_sql_query,
            explanation=explanation,
            results=final_results,
            execution_time=execution_time,
            tokens=total_tokens
        ) 