from aiohttp import ClientError
from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.s3 import S3
from pandas import DataFrame
from os import path
import boto3

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_s3(data_dir: str, **kwargs) -> None:
    """
    Template for exporting data to a S3 bucket.
    Specify your configuration settings in 'io_config.yaml'.

    Docs: https://docs.mage.ai/design/data-loading#s3
    """
    config_path = path.join(get_repo_path(), 'env/config.yaml')
    config_profile = 'default'
    
    config_file_loader = ConfigFileLoader(config_path, config_profile)

    bucket_name = 'mlops-zoomcamp-2023-xharko-cekovski'
    object_key = "test"
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config_file_loader.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=config_file_loader.get('AWS_SECRET_ACCESS_KEY')
    )
    
    try:
        response = s3_client.upload_file(Filename=data_dir, Bucket=bucket_name, Key=object_key)
    except ClientError as e:
        print(e)
        return False
    return True

