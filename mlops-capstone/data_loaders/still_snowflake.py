import io
import os
import pandas as pd
import requests
from zipfile import ZipFile
import shutil
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """
    url = r'https://trash-ai-public.s3.us-west-1.amazonaws.com/tacotrashdataset.zip'
    if not os.path.exists("downloads"):
        os.mkdir("downloads")
    target = f"downloads/tacotrashdataset.zip"
    if os.path.isfile("downloads/tacotrashdataset.zip"):
            print("Already downloaded")
    else:
        with requests.get(url, stream=True) as r:
            # check header to get content length, in bytes
            total_length = int(r.headers.get("Content-Length"))

            # save the output to a file
            with open(target, 'wb') as output:
                shutil.copyfileobj(r.raw, output)
    
    extract_data(target, *args)

    return target

def extract_data(target, *args):
    if not os.path.exists("downloads/dataset"):
        os.mkdir("downloads/dataset")
    
    if os.path.isfile("downloads/dataset/data/batch_1/000000.jpg"):
        print("Files already extracted")
    else:
        with ZipFile(target, 'r') as zObject:
        
            # Extracting all the members of the zip 
            # into a specific location.
            zObject.extractall(path="downloads/dataset")
    print("Extraction finished")

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    download_success = os.path.isfile(output)
    assert download_success is not False, 'The output is undefined'
