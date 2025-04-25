import os
import boto3
import shutil
from botocore import UNSIGNED
from botocore.client import Config

def sync_s3_to_documents(bucket_name, aws_access_key=None, aws_secret_key=None, 
                         aws_region='us-east-1', public_bucket=False):
    """
    Creates a 'documents' directory (or empties it if it exists) and populates it
    with files from the specified S3 bucket.
    
    Args:
        bucket_name (str): Name of the S3 bucket to download files from
        aws_access_key (str, optional): AWS access key. If None and public_bucket=False, 
                                       uses environment credentials.
        aws_secret_key (str, optional): AWS secret key. If None and public_bucket=False, 
                                       uses environment credentials.
        aws_region (str, optional): AWS region. Defaults to 'us-east-1'.
        public_bucket (bool, optional): If True, treats the bucket as public. Defaults to False.
        
    Returns:
        bool: True if successful, False if error occurred
    """
    try:
        # Set up the S3 client
        if public_bucket:
            # Create an unsigned config for public bucket access
            s3_client = boto3.client(
                's3',
                region_name=aws_region,
                config=Config(signature_version=UNSIGNED)
            )
        elif aws_access_key and aws_secret_key:
            # Use provided credentials
            s3_client = boto3.client(
                's3',
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            # Use environment credentials
            s3_client = boto3.client('s3', region_name=aws_region)
        
        # Create documents directory path
        documents_dir = os.path.join(os.getcwd(), 'documents')
        
        # If documents directory exists, remove all its contents
        if os.path.exists(documents_dir):
            shutil.rmtree(documents_dir)
        
        # Create the documents directory
        os.makedirs(documents_dir)
        
        # Try to list objects in the bucket
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name)
            
            # Download each file
            if 'Contents' in response:
                for item in response['Contents']:
                    file_key = item['Key']
                    # Skip directories/folders
                    if not file_key.endswith('/'):
                        # Create local path
                        local_file_path = os.path.join(documents_dir, os.path.basename(file_key))
                        # Download the file
                        s3_client.download_file(bucket_name, file_key, local_file_path)
                        print(f"Downloaded: {file_key}")
                
                print(f"Successfully synced all files from {bucket_name} to documents directory")
                return True
            else:
                print(f"No files found in bucket {bucket_name}")
                return True
                
        except Exception as first_error:
            # If first attempt failed, try alternate access method for public buckets
            if public_bucket or first_error.__class__.__name__ == 'NoCredentialsError':
                print(f"First attempt failed: {str(first_error)}")
                print("Trying alternate method for public bucket access...")
                
                # For public buckets, try generating URLs and downloading directly
                base_url = f"https://{bucket_name}.s3.amazonaws.com/"
                
                # Try a few common files
                common_files = [
                    "index.html", "data.csv", "README.txt", "info.json",
                    "document.pdf", "image.jpg", "file.txt"
                ]
                
                files_downloaded = 0
                for filename in common_files:
                    file_url = base_url + filename
                    local_path = os.path.join(documents_dir, filename)
                    
                    try:
                        # Configure the S3 client for public file access
                        s3_client.download_file(
                            bucket_name, 
                            filename, 
                            local_path
                        )
                        print(f"Downloaded: {filename}")
                        files_downloaded += 1
                    except Exception:
                        # Skip files that don't exist
                        pass
                
                if files_downloaded > 0:
                    print(f"Successfully downloaded {files_downloaded} files using public access")
                    return True
                else:
                    raise Exception("Could not access bucket with either method")
            else:
                # If not trying public access, re-raise the original error
                raise first_error
            
    except Exception as e:
        print(f"Error syncing files from S3: {str(e)}")
        print("If this is a public bucket, try calling the function with public_bucket=True")
        print("If credentials are required, ensure they are correct and have sufficient permissions")
        return False