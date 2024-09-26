import random
import boto3
import json
from image_data import metadata

# AWS S3 Configuration
S3_BUCKET = 'com.eqbridges.mnist-img-archive'
REGION = 'us-east-1'

# Function to generate pre-signed S3 URLs
def generate_s3_presigned_url(s3_client, bucket, key, expiration=3600):
    return s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket,
            'ResponseContentType': 'image/png', 'Key': key}, ExpiresIn=expiration)

def get_random_image_urls(n=100):
    """
    Returns n random image URLs from the bucket.
    """
    # Ensure n is in range of 1:100
    if n <= 0:
        n=1
    n = min(n, 100)

    # Select n random entries from the metadata
    random_metadata = random.sample(metadata, n)

    # Initialize S3 client
    s3 = boto3.client('s3', region_name=REGION)

    # Generate pre-signed URLs based on the selected metadata
    image_urls = [ generate_s3_presigned_url(s3, S3_BUCKET, entry) for entry in random_metadata ]

    return image_urls

# Lambda handler function
def lambda_handler(event, context):
    n = int(event.get('queryStringParameters', {}).get('n', 100))  # Default to 100 if n is not provided
    data = get_random_image_urls(n)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type': 'application/json"
        },
        "body": json.dumps(data)
    }

# For testing locally
if __name__ == '__main__':
    random_image_urls = get_random_image_urls(n=10)
    for image in random_image_urls:
        print(image)
