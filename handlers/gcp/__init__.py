import os
from google.oauth2 import service_account
from handlers.gcp.big_query import BigQuery

CREDENTIALS = service_account.Credentials.from_service_account_file(os.environ["SERVICE_ACCOUNT_PATH"])
BQ_CLIENT = BigQuery.initialize_bigquery_client(credentials=CREDENTIALS, project='card-ai-389220')