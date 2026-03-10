import os
import tiktoken
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

_EMBEDDINGS_DATASET = "vector_embeddings"
_EMBEDDINGS_TABLE   = "example_vectors"

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


class DailSQL:
    def __init__(self, llm_client, model_name, schema_str, embedding_model,
                 bq_client, project_id, dataset_name, use_skeleton=True, k=3):
        self.llm_client = llm_client
        self.model_name = model_name
        self.schema_str = schema_str
        self.embedding_model = embedding_model
        self.bq_client = bq_client
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.use_skeleton = use_skeleton
        self.k = k

    @staticmethod
    def initialize_agent(llm_client, model_name, project_id, dataset_name,
                         embeddings_dir='dail-sql', use_skeleton=True, k=3):
        bq_client = bigquery.Client(project=project_id)
        embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        DailSQL._ensure_embeddings(bq_client, project_id, embedding_model, embeddings_dir)
        schema_str = DailSQL._export_schema(bq_client, project_id, dataset_name)
        return DailSQL(
            llm_client=llm_client,
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

        # Ensure the dataset exists
        dataset_ref = bigquery.Dataset(f"{project_id}.{_EMBEDDINGS_DATASET}")
        try:
            bq_client.create_dataset(dataset_ref)
        except Exception:
            pass  # already exists

        # Load from pre-built parquet if available, otherwise embed from jsonl
        parquet_path = os.path.join(embeddings_dir, "example_vectors.parquet")
        if os.path.exists(parquet_path):
            print(f"Loading embeddings from {parquet_path}")
            df = pd.read_parquet(parquet_path)
        else:
            print("Parquet not found — generating embeddings from jsonl files...")
            df = pd.read_json(os.path.join(embeddings_dir, "examples_full.jsonl"), lines=True)
            df["skeleton"] = pd.read_json(
                os.path.join(embeddings_dir, "skeletons.jsonl"), lines=True
            )["sql"]
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

        resp = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        sql_query = self._parse_sql(resp.choices[0].message.content)

        try:
            results = self.bq_client.query(sql_query).result()
            exec_results = [dict(row) for row in results]
        except Exception as e:
            print(f"SQL execution failed: {e}")
            exec_results = []

        # No NL answer generation in DAIL-SQL; return empty string
        return sql_query, exec_results, "", input_tokens
