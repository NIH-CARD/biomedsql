import tiktoken
from tabulate import tabulate
import pandas as pd

def format_schema(schema):
    rows = []
    for field in schema:
        rows.append([f"- Name: {field.name}", f"Type: {field.field_type}", f"Mode: {field.mode}"])
    return "\n".join(["  " + " | ".join(row) for row in rows])

def format_table_sample(rows):
    if rows.total_rows == 0:
        return "(No data)"
    headers = rows.schema
    table = [[row.get(field.name) for field in headers] for row in rows]
    col_names = [field.name for field in headers]
    return tabulate(table, headers=col_names, tablefmt="grid")

def create_table_info(bq_handler, dataset_id, num_rows):
    output = []

    output.append(f"Tables in dataset {dataset_id}")

    tables = bq_handler.get_tables(dataset_id)

    table_names = [table.table_id for table in tables]

    output += table_names

    output.append(f"\nSchema details for tables in dataset {dataset_id}\n")

    for table in tables:
        table_id = f"{dataset_id}.{table.table_id}"
        output.append(f"Table: {table.table_id}")
        output.append("Schema:")
        schema = bq_handler.get_table_schema(table_id)
        output.append(format_schema(schema))

        if num_rows > 0:
            output.append("\nRandomly sampled rows from table:")
            rows = bq_handler.sample_rows(table_id, num_rows=num_rows)
            output.append(format_table_sample(rows))
            output.append("\n")

    text_output = "\n".join(output)

    return text_output

def get_examples(example_query_prompt, num_examples):
    if num_examples > 0 and num_examples < 5:
        return example_query_prompt.split(f'Example {num_examples+1}')[0]
    elif num_examples == 5:
        return example_query_prompt
    else:
        return ''

def get_thresholds(thresholds):
    if thresholds:
        threshold_rules = "Use the following p-value thresholds for questions about statistical significance:\n    1. p < 5e-8 for genome-wide significance\n    2. p_SMR < 2.95e-6, p_HEIDI > 0.01 for SMR significance"
        return threshold_rules
    else:
        return ""

def read_prompts(prompt_dir):
    with open(f'{prompt_dir}/sql_generation_prompt.txt', 'r') as f:
        sql_gen_prompt = f.read()
        f.close()

    with open(f'{prompt_dir}/example_queries.txt', 'r') as f:
        example_query_prompt = f.read()
        f.close()

    with open(f'{prompt_dir}/natural_language_answer_prompt.txt', 'r') as f:
        nl_answer_prompt = f.read()
        f.close()

    with open(f'{prompt_dir}/bioscore_prompt.txt', 'r') as f:
        bioscore_prompt = f.read()
        f.close()

    return sql_gen_prompt, example_query_prompt, nl_answer_prompt, bioscore_prompt

def count_tokens_tiktoken(string: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    string = str(string) if string is not None else ""
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_sql(llm_handler, model_name, prompt):
    input_tokens = count_tokens_tiktoken(prompt)

    try:
        response = llm_handler.query(
            model_name = model_name,
            max_tokens = 1024,
            temperature = 0,
            query_text = prompt
        )
        return response, input_tokens, 1

    except Exception as e:
        print(f"LLM call error: {e}")
        return "", input_tokens, 0
    
def parse_sql_query(query):
    try: 
        if '```sql' in query:
            start_index = query.find("```sql") + len("```sql")
            end_index = query.find("```", start_index)
            if end_index != -1:
                query = query[start_index:end_index].strip()
                return query, 1
        elif '```' in query:
            start_index = query.find("```") + len("```")
            end_index = query.find("```", start_index)
            if end_index != -1:
                query = query[start_index:end_index].strip()
                return query, 1
        print('Error parsing SQL query')
        return "", 0
    except Exception as e:
        print('Error parsing SQL query')
        return "", 0

def run_sql(bq_handler, query):
    try:
        result = bq_handler.query_db(
            query=query
        )
        execution_results = [dict(row) for row in result]
        return execution_results, 1
    except Exception as e:
        print(f'Failed to run query: {query}')
        return [], 0
    
def generate_answer(llm_handler, model_name, prompt):
    input_tokens = count_tokens_tiktoken(prompt)

    try:
        response = llm_handler.query(
            model_name = model_name,
            max_tokens = 1024,
            temperature = 0,
            query_text = prompt
        )
        return response, input_tokens, 1

    except Exception as e:
        print(f"LLM call error: {e}")
        return "", input_tokens, 0
        
def bioscore_components(llm_handler, model_name, prompt):
    input_tokens = count_tokens_tiktoken(prompt)

    try:
        response = llm_handler.query(
            model_name = model_name,
            max_tokens = 1024,
            temperature = 0,
            query_text = prompt
        )
        return response, input_tokens

    except Exception as e:
        print(f"LLM call error: {e}")
        return "0", input_tokens