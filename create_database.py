import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from google.cloud import storage, bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import Conflict

def download_db_data(local_dir):
    local_path = snapshot_download(
        repo_id="NIH-CARD/BiomedSQL",
        repo_type="dataset",
        allow_patterns="db_data/*",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    save_dir = f'{local_path}/db_data'
    print(f'Files saved to {save_dir}')

    return save_dir

def upload_to_bigquery(data_dir):
    ### TODO: pull from .env when experiments finish
    ### TODO: add an option to go straight from local
    CREDENTIALS = service_account.Credentials.from_service_account_file(os.environ["SERVICE_ACCOUNT_PATH"])
    project = os.environ["PROJECT_ID"]
    storage_client = storage.Client(credentials=CREDENTIALS, project=project)
    bq_client = bigquery.Client(credentials=CREDENTIALS, project=project)
    bucket_name = os.environ["BUCKET_NAME"]
    location = os.environ["LOCATION"]
    dataset_name = os.environ["DATASET_NAME"]

    dataset_id = f"{project}.{dataset_name}"

    try:
        bucket = storage_client.create_bucket(bucket_name, location=location)
    except Conflict:
        print(f'Bucket already exists')
        bucket = storage_client.get_bucket(bucket_name)

    try:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = location

        bq_client.create_dataset(dataset)
    except Conflict:
        bq_client.get_dataset(dataset_id)

    for file in os.listdir(data_dir):
        blob = bucket.blob(file)

        if not blob.exists():
            blob.upload_from_filename(f'{data_dir}/{file}')
            print(f'File {file} uploaded!')
        else:
            print(f'File {file} already in bucket!')

        file_name = file.split('.')[0]

        table_id = f'{dataset_id}.{file_name}'
        uri = f'gs://{bucket_name}/{file}'
        
        job_config = bigquery.LoadJobConfig(
            source_format = bigquery.SourceFormat.PARQUET,
            autodetect = True,
            write_disposition = "WRITE_TRUNCATE"
        )

        load_job = bq_client.load_table_from_uri(uri, table_id, job_config=job_config)
        load_job.result() 
 
def main():
    load_dotenv('config/.env')

    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    db_data_dir = download_db_data(data_dir)

    upload_to_bigquery(db_data_dir)

if __name__ == '__main__':
    main()