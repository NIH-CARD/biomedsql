import os
import re
import json
import tiktoken
import sqlparse
import pandas as pd
from dataclasses import dataclass, field
from typing import Any
from datasets import load_dataset
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from typeguard import typechecked
from handlers.llms.base_llm import BaseLLM
from utils.experiments_utils import substitute_env_placeholders
from handlers.llms import AZURE_CLIENT, GEMINI_CLIENT, ANTHROPIC_CLIENT
from handlers.llms.azure_openai_llm import AzureOpenAILLM
from handlers.llms.gemini_llm import GeminiLLM
from handlers.llms.anthropic_llm import AnthropicLLM

_EMBEDDINGS_DATASET = "vector_embeddings"
_EMBEDDINGS_TABLE   = "example_vectors"

# Skeleton generation — replaces literals with '?' for example retrieval
_LITERAL_RE = re.compile(
    r"""
        (?:'[^']*' | "[^"]*")            # quoted strings
      | (?<![\w-])\d+(?:\.\d+)?(?![\w]) # stand-alone numbers
      | \b(?:true|false|null)\b          # booleans / NULL
    """,
    re.IGNORECASE | re.VERBOSE,
)

PROMPT_TMPL = """You are an expert BigQuery SQL generator. Return only the SQL query with no explanation.

Use these guidelines when generating the query:
    1. Review the database schema.
    2. Review the user's question.
    3. Generate a valid Google BigQuery SQL query that answers the question based on the schema.
    4. Always enclose table references in backticks, e.g. `project.dataset.table`.
    5. Make use of BigQuery-specific functions and syntax where appropriate (e.g., DISTINCT, aliases, ORDER BY).
    6. Always include the UUID column in your SELECT statements, except in cases of questions where the COUNT and GROUP BY functions are needed.
    7. Unless the user explicitly requests a different LIMIT, default your queries to LIMIT 100.
    8. Output ONLY the raw SQL query (no additional commentary or explanations).
    9. Avoid SELECT *; select only the necessary columns to answer the user's query.
    10. Ensure that any disease names that contain an apostrophe in the query are surrounded by double quotes (e.g., "Alzheimer's Disease").

### Database Name
{project_id}.{dataset_name}

### Database Schema
{schema}

### Relevant Examples
{examples}

### Question
{question}
SQL:"""


def _to_skeleton(sql: str) -> str:
    no_literals = _LITERAL_RE.sub("?", sql)
    return sqlparse.format(no_literals, keyword_case="upper", strip_comments=True, reindent=False)


def _write_jsonl(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json.dump({"id": row["id"], "q": row["q"], "sql": row["sql"]}, f, ensure_ascii=False)
            f.write("\n")


@typechecked
@dataclass(frozen=True)
class DailSQL:
    llm_handler: BaseLLM
    model_name: str
    schema_str: str
    embedding_model: Any  # SentenceTransformer
    bq_client: Any        # bigquery.Client
    project_id: str
    dataset_name: str
    use_skeleton: bool = True
    k: int = 3

    @staticmethod
    def initialize_agent(model_provider, model_name, project_id, dataset_name,
                         embeddings_dir='dail-sql', use_skeleton=True, k=3):
        if model_provider == 'azure_openai':
            llm_handler = AzureOpenAILLM(AZURE_CLIENT)
        elif model_provider == 'gemini':
            llm_handler = GeminiLLM(GEMINI_CLIENT)
        elif model_provider == 'anthropic':
            llm_handler = AnthropicLLM(ANTHROPIC_CLIENT)
        else:
            raise ValueError(f'Invalid provider for dail interaction: {model_provider}')

        bq_client = bigquery.Client(project=project_id)
        embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        DailSQL._ensure_embeddings(bq_client, project_id, embedding_model, embeddings_dir)
        schema_str = DailSQL._export_schema(bq_client, project_id, dataset_name)
        return DailSQL(
            llm_handler=llm_handler,
            model_name=model_name,
            schema_str=schema_str,
            embedding_model=embedding_model,
            bq_client=bq_client,
            project_id=project_id,
            dataset_name=dataset_name,
            use_skeleton=use_skeleton,
            k=k,
        )

    @staticmethod
    def _ensure_embeddings(bq_client, project_id, embedding_model, embeddings_dir):
        """Check if the vector embeddings table exists in BigQuery; create it if not."""
        table_id = f"{project_id}.{_EMBEDDINGS_DATASET}.{_EMBEDDINGS_TABLE}"
        try:
            bq_client.get_table(table_id)
            print(f"Embeddings table found: {table_id}")
            return
        except NotFound:
            print(f"Embeddings table not found. Building from {embeddings_dir}/...")

        dataset_ref = bigquery.Dataset(f"{project_id}.{_EMBEDDINGS_DATASET}")
        try:
            bq_client.create_dataset(dataset_ref)
        except Exception:
            pass

        parquet_path = os.path.join(embeddings_dir, "example_vectors.parquet")
        if os.path.exists(parquet_path):
            print(f"Loading embeddings from {parquet_path}")
            df = pd.read_parquet(parquet_path)

        else:
            examples_path  = os.path.join(embeddings_dir, "examples_full.jsonl")
            skeletons_path = os.path.join(embeddings_dir, "skeletons.jsonl")

            if not os.path.exists(examples_path) or not os.path.exists(skeletons_path):
                train_path = os.path.join(embeddings_dir, "train.csv")
                if not os.path.exists(train_path):
                    print(f"train.csv not found. Downloading from HuggingFace...")
                    os.makedirs(embeddings_dir, exist_ok=True)
                    hf = load_dataset(
                        "csv",
                        data_files="https://huggingface.co/datasets/NIH-CARD/BiomedSQL/resolve/main/benchmark_data/train.csv"
                    )
                    substitute_env_placeholders(hf["train"].to_pandas()).to_csv(train_path, index=False)
                print(f"Generating example files from {train_path}...")
                df_train = pd.read_csv(train_path).sample(n=40, random_state=42)
                df_train["skeleton"] = df_train["benchmark_query"].apply(_to_skeleton)
                df_train = df_train.rename(
                    columns={"uuid": "id", "question": "q", "benchmark_query": "sql"}
                )
                _write_jsonl(df_train[["id", "q", "sql"]], examples_path)
                _write_jsonl(
                    df_train[["id", "q", "skeleton"]].rename(columns={"skeleton": "sql"}),
                    skeletons_path,
                )

            print("Generating embeddings from jsonl files...")
            df = pd.read_json(examples_path, lines=True)
            df["skeleton"] = pd.read_json(skeletons_path, lines=True)["sql"]
            df["embedding"] = embedding_model.encode(
                df["q"].tolist(), normalize_embeddings=True, batch_size=64
            ).tolist()
            df["skeleton_embedding"] = embedding_model.encode(
                df["sql"].tolist(), normalize_embeddings=True, batch_size=64
            ).tolist()
            df.to_parquet(parquet_path)

        job = bq_client.load_table_from_dataframe(
            df[["id", "q", "sql", "skeleton", "embedding", "skeleton_embedding"]],
            table_id,
        )
        job.result()
        print(f"Embeddings uploaded to {table_id}")

    @staticmethod
    def _export_schema(bq_client, project_id, dataset_name):
        q = f"""
        SELECT table_name, column_name, data_type
        FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
        ORDER BY table_name, ordinal_position
        """
        rows = bq_client.query(q).result()
        schema = defaultdict(list)
        for r in rows:
            schema[r.table_name].append(f"{r.column_name} {r.data_type}")
        return "\n".join(f"{tbl}({', '.join(cols)})" for tbl, cols in schema.items())

    def _retrieve_examples(self, question):
        tbl = f"{self.project_id}.{_EMBEDDINGS_DATASET}.{_EMBEDDINGS_TABLE}"
        q_vec = self.embedding_model.encode([question], normalize_embeddings=True)[0].tolist()
        sql = f"""
            SELECT id, q, sql, skeleton,
            COSINE_DISTANCE(embedding, @v) AS hybrid_score
            FROM `{tbl}`
            ORDER BY hybrid_score DESC
            LIMIT {self.k}
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("v", "FLOAT64", q_vec)]
        )
        return list(self.bq_client.query(sql, job_config=cfg))

    def _build_prompt(self, question, ex_rows):
        if self.use_skeleton:
            ex_text = "\n\n".join(f"-- {r.q}\n{r.skeleton}" for r in ex_rows)
        else:
            ex_text = "\n\n".join(f"-- {r.q}\n{r.sql}" for r in ex_rows)
        return PROMPT_TMPL.format(
            project_id=self.project_id,
            dataset_name=self.dataset_name,
            schema=self.schema_str,
            examples=ex_text,
            question=question,
        )

    @staticmethod
    def _parse_sql(query):
        if '```sql' in query:
            start = query.find("```sql") + len("```sql")
            end = query.find("```", start)
            if end != -1:
                return query[start:end].strip()
        elif '```' in query:
            start = query.find("```") + len("```")
            end = query.find("```", start)
            if end != -1:
                return query[start:end].strip()
        return query.strip()

    @staticmethod
    def _count_tokens(string, model="gpt-4o"):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(str(string) if string is not None else ""))

    def run_agent(self, question):
        examples = self._retrieve_examples(question)
        prompt = self._build_prompt(question, examples)
        input_tokens = self._count_tokens(prompt)

        sql_raw = self.llm_handler.query(
            model_name=self.model_name,
            max_tokens=4096,
            temperature=0,
            query_text=prompt,
        )
        sql_query = self._parse_sql(sql_raw)

        try:
            results = self.bq_client.query(sql_query).result()
            exec_results = [dict(row) for row in results]
        except Exception as e:
            print(f"SQL execution failed: {e}")
            exec_results = []

        # NL answer generation in pipeline since its not "built in" like other interaction paradigms
        return sql_query, exec_results, "", input_tokens
