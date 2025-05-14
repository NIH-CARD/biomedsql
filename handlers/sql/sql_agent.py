from dataclasses import dataclass
from typeguard import typechecked
from handlers.sql.sql_handlers import SQLAPI

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

    sql_api: SQLAPI
    max_retries: int = 3
    
    @typechecked
    async def run_agent(self, subquery: str, num_passes: int = 1):
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
                    print(subquery)

                    # Step 1: Identify relevant columns
                    relevant_cols_response, relevant_cols_tokens = await self.sql_api.get_relevant_columns(subquery)
                    total_tokens += relevant_cols_tokens

                    # Step 2: Generate initial SQL query
                    general_query_text, general_query_tokens = await self.sql_api.generate_sql_query(
                        subquery, 
                        relevant_cols_response
                    )
                    total_tokens += general_query_tokens
                    general_query = await self.sql_api.extract_sql_code(general_query_text)

                    # Step 3: Execute with up to self.max_retries
                    general_results = []
                    execution_failed = False
                    for attempt in range(self.max_retries):
                        results, fail_flag = await self.sql_api.execute_sql_query(general_query)
                        if not fail_flag:
                            general_results = results
                            break
                        else:
                            corrected_text, check_tokens = await self.sql_api.sql_query_checker(
                                subquery,
                                relevant_cols_response,
                                general_query,
                                str(results)
                            )
                            total_tokens += check_tokens

                            general_query = await self.sql_api.extract_sql_code(corrected_text) or general_query
                            if attempt == self.max_retries - 1:
                                execution_failed = True

                    # if execution_failed:
                    #     answer = "SQL execution failed. Therefore there is no relevant information to answer the question."
                    #     return general_query, general_results, '',  [], answer, total_tokens

                    # Step 4: Optionally refine query
                    refined_query, refined_tokens = await self.sql_api.generate_refined_sql(
                        question=subquery,
                        sql_query=general_query,
                        result=general_results
                    )
                    total_tokens += refined_tokens

                    # Step 5: Execute the refined query
                    refined_results, refined_failed = await self.sql_api.execute_sql_query(refined_query)
                    
                    if execution_failed and refined_failed:
                        english_response = "SQL execution failed. Therefore there is no relevant information to answer the question."
                        # refined_results = []
                    
                    else:
                        # Step 6: Aggregated English response (the final LLM call)
                        english_response, agg_tokens = await self.sql_api.aggregated_response(
                            subquery,
                            general_query,
                            refined_query,
                            general_results,
                            refined_results
                        )
                        total_tokens += agg_tokens

                    print(english_response)

                    sufficient_response, retry_tokens = await self.sql_api.sufficient_response(
                        subquery,
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
            error_str = f"Error in run_agent for '{subquery}': {str(e)}"
            print(f"Error: {error_str}")
            answer = "SQL execution failed. Therefore there is no relevant information to answer the question."
            return '', [], '', [], answer, total_tokens