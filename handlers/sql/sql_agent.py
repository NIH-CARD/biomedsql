from dataclasses import dataclass
from typeguard import typechecked
from handlers.sql.sql_handlers import SQLHandler

@dataclass(frozen=True)
class SQLAgent():
    """
    This agent uses a two-turn approach to:
    - Determine which columns/tables are relevant
    - Generate an initial BigQuery SQL query
    - Refine & correct it if it fails
    - Return the final LLM response as the knowledge tuple

    The final answer is the direct result from the last LLM call in the pipeline.
    """

    sql_handler: SQLHandler
    max_retries: int = 3
    
    @typechecked
    def run_agent(self, question: str, num_passes: int = 1):
        """
        The main method that:
        1) Identifies relevant columns
        2) Generates initial query
        3) Executes & refines up to 'max_retries'
        4) Possibly a final refinement
        5) Returns final LLM response in KnowledgeTuple
        """
        try:
            total_tokens = 0
            sufficient_response = 'no'

            for i in range(num_passes):
                if sufficient_response.lower() == 'no':
                    print(question)

                    # Step 1: Identify relevant columns
                    relevant_cols_response, relevant_cols_tokens = self.sql_handler.get_relevant_columns(question)
                    total_tokens += relevant_cols_tokens

                    # Step 2: Generate initial SQL query
                    general_query_text, general_query_tokens = self.sql_handler.generate_sql_query(
                        question, 
                        relevant_cols_response
                    )
                    total_tokens += general_query_tokens
                    general_query = self.sql_handler.extract_sql_code(general_query_text)

                    # Step 3: Execute with up to self.max_retries
                    general_results = []
                    execution_failed = False
                    for attempt in range(self.max_retries):
                        results, fail_flag = self.sql_handler.execute_sql_query(general_query)
                        if not fail_flag:
                            general_results = results
                            break
                        else:
                            corrected_text, check_tokens = self.sql_handler.sql_query_checker(
                                question,
                                relevant_cols_response,
                                general_query,
                                str(results)
                            )
                            total_tokens += check_tokens

                            general_query = self.sql_handler.extract_sql_code(corrected_text) or general_query
                            if attempt == self.max_retries - 1:
                                execution_failed = True

                    # Step 4: Optionally refine query
                    refined_query, refined_tokens = self.sql_handler.generate_refined_sql(
                        question=question,
                        sql_query=general_query,
                        result=general_results
                    )
                    total_tokens += refined_tokens

                    # Step 5: Execute the refined query
                    refined_results, refined_failed = self.sql_handler.execute_sql_query(refined_query)
                    
                    if execution_failed and refined_failed:
                        english_response = "SQL execution failed. Therefore there is no relevant information to answer the question."
                        # refined_results = []
                    
                    else:
                        # Step 6: Aggregated English response (the final LLM call)
                        english_response, agg_tokens = self.sql_handler.aggregated_response(
                            question,
                            general_query,
                            refined_query,
                            general_results,
                            refined_results
                        )
                        total_tokens += agg_tokens

                    print(english_response)

                    sufficient_response, retry_tokens = self.sql_handler.sufficient_response(
                        question,
                        general_query,
                        refined_query,
                        general_results,
                        refined_results,
                        english_response
                    )
                    total_tokens += retry_tokens
                    print(f'Is the answer sufficient?: {sufficient_response.strip()}')
                else:
                    return general_query, general_results, refined_query, refined_results, english_response, total_tokens
        
            return general_query, general_results, refined_query, refined_results, english_response, total_tokens

        except Exception as e:
            error_str = f"Error in run_agent for '{question}': {str(e)}"
            print(f"Error: {error_str}")
            answer = "SQL execution failed. Therefore there is no relevant information to answer the question."
            return '', [], '', [], answer, total_tokens