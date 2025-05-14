import os
from handlers.gcp.big_query import BigQuery

BQ_CLIENT = BigQuery.initialize_bigquery_client(credentials=os.environ["CREDENTIALS"], project='card-ai-389220')