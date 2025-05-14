import re
import json
import tiktoken
from dataclasses import dataclass
from typeguard import typechecked
from typing import Any, Dict
from handlers.llms.base_llm import BaseLLM
from handlers.gcp.big_query import BigQuery

from handlers.sql import PROJECT_ID, DATASET_NAME, QUERY_LIMIT

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
class SQLAPI:
    """
    A class to interact with SQL agent resources.
    This class encapsulates the logic for running SQL queries.
    """
    table_info: str
    table_info_concise: str
    llm: BaseLLM
    llm_query_params: Dict[str, Any]
    bq_handler: BigQuery

    async def execute_sql_query(self, query: str):
        """Execute a SQL query on BigQuery and return (results, failed_flag)."""
        execution_failed = False
        try:
            results = self.bq_handler.query_db(query)
            rows = [dict(row) for row in results]
            print(f"Query executed successfully. Returning {len(rows)} rows.")
            return rows, execution_failed
        except Exception as e:
            print(f"An error occurred: {e}")
            execution_failed = True
            return [], execution_failed

    async def get_relevant_columns(self, question: str):
        """
        Calls AzureOpenAI to parse the schema and figure out which columns might be relevant.
        """
        prompt = f"""
        You are a BioMedical Domain Expert with deep database knowledge. You have the following database schema:

        {self.table_info_concise}

        The user has asked a question about this biomedical data:

        "{question}"

        Your task: 
        1. Identify the single table or multiple tables (if absolutely necessary) that would provide 
        the *full* answer to this question.
        2. From these table(s), list *all columns* that might be relevant to fully answer the question. 
        (Because a downstream aggregator will handle details, do NOT omit columns that may be relevant.)

        Format your response **strictly** as:
        TABLE_NAME: col1, col2, col3, ...

        - Provide no extra commentary or text.
        - If multiple tables are truly needed, list each in a new line, in the same format.
        """
        tokens = count_tokens_tiktoken(prompt)

        response = await self.llm.query_async(
            model_name=self.llm_query_params['model'],
            max_tokens=self.llm_query_params['max_tokens'],
            temperature=self.llm_query_params['temperature'],
            query_text=prompt
        )
        
        return response, tokens

    async def sql_query_checker(
        self,
        question: str,
        relevant_columns: str,
        general_query: str,
        general_results: str
    ):
        """
        If a query fails, ask the LLM to fix it.
        """
        prompt = f"""
        You are a SQL debugging assistant for Google BigQuery. Below is the database schema, 
        the failed query, and the error message or unexpected results:

        === DATABASE SCHEMA START ===
        {self.table_info}
        === DATABASE SCHEMA END ===

        === FAILED SQL QUERY START ===
        ```sql
        {general_query}
        ```
        === FAILED SQL QUERY END ===

        === ERROR OR RESULTS START ===
        {general_results}
        === ERROR OR RESULTS END ===

        The user originally asked:
        "{question}"

        Relevant columns identified for answering this question:
        {relevant_columns}

        Your task:
        - Analyze the failed query and the error or result details.
        - Generate a corrected SQL query that resolves the issue, 
        ensuring it's correct for BigQuery and fits the schema.
        - Format the corrected query as a valid SQL query in a markdown fenced block:

        ```sql
        SELECT ...
        FROM ...
        ```
        """
        tokens = count_tokens_tiktoken(prompt)

        response = await self.llm.query_async(
            model_name=self.llm_query_params['model'],
            max_tokens=self.llm_query_params['max_tokens'],
            temperature=self.llm_query_params['temperature'],
            query_text=prompt
        )
        
        return response.strip(), tokens

    async def generate_sql_query(
        self,
        question: str,
        relevant_columns: str
    ):
        """
        Generate a single valid BigQuery SQL query, referencing the 'PROJECT_ID', 'DATASET_NAME',
        user question, relevant columns, etc.
        """
        prompt = f"""
        You are a highly proficient BigQuery SQL generator in the biomedical domain.

        Database schema:
        {self.table_info}

        The user asked:
        "{question}"

        Previously identified relevant columns/tables:
        {relevant_columns}

        Instructions:
        - Generate exactly one valid BigQuery SQL query that retrieves all relevant columns 
        from the relevant_columns list.
        - Do not filter out p-values, do not do advanced thresholds unless the user explicitly stated them.
        - If user mentions FDA approval, include those columns. 
        - If user mentions allele frequencies, include effect and non-effect allele freq columns.
        - FROM clause: `{PROJECT_ID}.{DATASET_NAME}.TABLE_NAME`
        - Always include the UUID column in your SELECT statements, except in cases of questions where the COUNT and ORDER BY functions are needed.
        - Unless the user explicitly requests a different LIMIT, default your queries to LIMIT 100.
        - Return only the final SQL in a markdown code block:
        ```sql
        SELECT ...
        FROM ...
        ```
        """
        tokens = count_tokens_tiktoken(prompt)

        response = await self.llm.query_async(
            model_name=self.llm_query_params['model'],
            max_tokens=self.llm_query_params['max_tokens'],
            temperature=self.llm_query_params['temperature'],
            query_text=prompt
        )
        
        return response.strip(), tokens

    async def extract_sql_code(self, text: str):
        """
        Extract the SQL from a ```sql ... ``` block
        """
        pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        print("No SQL code block found in text.")
        return ""

    async def generate_refined_sql(
        self,
        question: str,
        sql_query: str,
        result: list
    ):
        """
        Possibly refine the query if the user might want additional filtering or thresholding,
        e.g. p-values, etc. If no refinement is needed, just return the original query.
        """
        # try:
        #     table_info = self.read_table_info_from_gcs(table_info_concise_path, bucket_name="sql_agent_bucket")
        # except:
        #     table_info = "(No table info found)"

        threshold_rules = """
        1) p < 5e-8 for genome-wide significance
        2) p_SMR < 2.95e-6, p_HEIDI > 0.01 for SMR significance
        3) For FDA approvals, see columns isApproved or yearOfFirstApproval
        4) Always limit to 10 unless user requests more
        """

        # Convert response to something the LLM can read easily
        resp_str = json.dumps(result, indent=2)

        prompt = f"""
        You are a skillful BigQuery SQL refiner. The user might want additional thresholds or 
        see if there's advanced filtering needed, e.g. p-values or FDA approvals.

        Original question: "{question}"

        The previously generated SQL query was:
        ```sql
        {sql_query}
        ```

        The query's results (showing up to 10 rows):
        {resp_str}

        Database partial schema:
        {self.table_info_concise}

        Known threshold rules:
        {threshold_rules}

        If no extra thresholds or filters are implied, keep the same query.
        Otherwise, produce a refined SQL with the new filters, 
        returning it in a markdown code block:
        ```sql
        SELECT ...
        FROM ...
        ```
        """
        tokens = count_tokens_tiktoken(prompt)

        response = await self.llm.query_async(
            model_name=self.llm_query_params['model'],
            max_tokens=self.llm_query_params['max_tokens'],
            temperature=self.llm_query_params['temperature'],
            query_text=prompt
        )
        
        refined_text = response.strip()
        
        # Extract code
        pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(pattern, refined_text, re.DOTALL)
        if match:
            return match.group(1).strip(), tokens
        return sql_query, tokens

    async def aggregated_response(
        self,
        question: str,
        sql_query_1: str,
        sql_query_2: str,
        result_1: list,
        result_2: list
    ):
        """
        The final LLM call that merges the results of the two queries 
        (initial & refined) into a short English answer.
        """
        prompt = f"""
        You are a BioMedical Domain expert that is returning a concise answer to the user's question 
        based on two sets of SQL queries and results. If not sure, say you do not know.

        Question: {question}

        SQL query 1: {sql_query_1}
        Result 1: {result_1}

        SQL query 2: {sql_query_2}
        Result 2: {result_2}
        """
        tokens = count_tokens_tiktoken(prompt)

        response = await self.llm.query_async(
            model_name=self.llm_query_params['model'],
            max_tokens=self.llm_query_params['max_tokens'],
            temperature=self.llm_query_params['temperature'],
            query_text=prompt
        )

        return response.strip(), tokens

    async def sufficient_response(
        self,
        question: str,
        sql_query_1: str,
        sql_query_2: str,
        result_1: list,
        result_2: list,
        answer: str
    ):
        prompt = f"""
        You are a biomedical domain and BigQuery expert that is determining if a text-to-SQL workflow should be run again.
        Based on the question SQL queries, their execution results, and the final answer, determine if you are confident in
        the answer. Use the following guidelines:
            1. If the SQL queries or answer contain errors, deem the answer as insufficient.
            2. If you have any doubts about the SQL queries, execution results, or answer, deem the answer as insufficient. 
            3. If there are any inconsistencies between the SQL queries, execution results, and answer, deem the answer as insufficient.
            4. Keep in mind that a negative answer (i.e. "No, ...") does not necessarily mean the answer is insufficient.
            5. Otherwise, use your best judgement. 
            6. Do not use any external information outside of what is provided.

        Question: {question}

        SQL query 1: {sql_query_1}
        Result 1: {result_1}

        SQL query 2: {sql_query_2}
        Result 2: {result_2}

        Answer: {answer}

        Please only return 'Yes' if the answer is sufficient and 'No' if it is insufficient.
        """

        tokens = count_tokens_tiktoken(prompt)

        response = await self.llm.query_async(
            model_name=self.llm_query_params['model'],
            max_tokens=self.llm_query_params['max_tokens'],
            temperature=self.llm_query_params['temperature'],
            query_text=prompt
        )

        return response, tokens