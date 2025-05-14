from abc import ABC
from dataclasses import dataclass
from typeguard import typechecked
from typing import Optional, List, Any
from google.cloud import bigquery
from google.oauth2 import service_account

@typechecked
@dataclass(frozen=True)
class BigQuery(ABC):
    """
    BigQuery is a class for handling GCP BigQuery operations.
    """

    bigquery_client: bigquery.Client

    @staticmethod
    def initialize_bigquery_client(
        credentials: Optional[service_account.Credentials] = None,
        project: Optional[str] = None
    ) -> Optional[bigquery.Client]:
        """
        Initialize BigQuery client.

        Parameters:
        - Optional[credentials] (service_account.Credentials): Credentials associated with the GCP.
        - Optional[project] (str): GCP name.


        Returns:
        - Optional[bigquery.Client]: The BigQuery client initialized with credentials (passed or default).
        """

        try:
            if credentials and project:
                return bigquery.Client(credentials=credentials, project=project)
            else:
                return bigquery.Client()
        except Exception as e:
            print(f'Error initializing BigQuery client: {e}')
            return None
        
    def upload_to_bq(self, gs_uri: str, dataset_id: str, table_name: str, file_type: Optional[str] = 'parquet') -> None:
        """
        Upload table to BigQuery.

        Parameters:
        - gs_url (str): Google storage object to be uploaded to BiqQuery
        - dataset_id (str): BigQuery dataset to upload table to.
        - table_name (str): ID to give the table within the BigQuery dataset
        - Optional[file_type] (str): File type of the object to be uploaded to BigQuery

        Returns:
        - None
        """
        
        table_id = f'{self.bigquery_client.project}.{dataset_id}.{table_name}'

        try:
            if file_type == 'parquet':
                load_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET)

            load_job = self.bigquery_client.load_table_from_uri(gs_uri, table_id, job_config=load_config)
            load_job.result()

            destination_table = self.bigquery_client.get_table(table_id)
            print("Loaded {} rows.".format(destination_table.num_rows))
        except Exception as e:
            print(f'Failed to load {gs_uri} to {table_id}: {e}')
    
    def query_db(self, query: str) -> Optional[Any]:
        """
        Run SQL query.

        Parameters:
        - query (str): SQL query to be run.

        Returns:
        - Any: Formatted execution results.
        """

        try:
            return self.bigquery_client.query(query).result()
        except Exception as e:
            print(f'Failed to run query: {query}')
            None
    
    def get_tables(self, dataset_id: str) -> List[Any]:
        """
        Get a list of tables in a dataset.

        Parameters:
        - dataset_id (str): Dataset to get tables from.

        Returns:
        - List[Any]: List of tables in the passed dataset.
        """

        try:
            dataset = self.bigquery_client.get_dataset(dataset_id)
            return list(self.bigquery_client.list_tables(dataset))
        except Exception as e:
            print(f'Error returning tables from {dataset_id}: {e}')
            return []
    
    def get_table_schema(self, table_id: str) -> List[Any]:
        """
        Get a tables schema.

        Parameters:
        - table_id (str): Table to get schema from.

        Returns:
        - List[Any]: Table schema.
        """

        try:
            table = self.bigquery_client.get_table(table_id)
            return table.schema
        except Exception as e:
            print(f'Error returning schame from {table_id}: {e}')
            return []
    
    def sample_rows(self, table_id: str, num_rows: Optional[int] = 5) -> Optional[Any]:
        """
        Sample a random set of rows from a table

        Parameters:
        - table_id (str): Table to sample from.
        - Optional[num_rows] (int): Number of rows to sample

        Returns
        - Optional[Any]: BigQuery execution results on success or None on failure.
        """

        try:
            query = f"SELECT * FROM `{table_id}` ORDER BY RAND() LIMIT {num_rows}"
            return self.query_db(query)
        except Exception as e:
            print(f'Error running query to sample {num_rows }rows from {table_id}: {e}')
            return None