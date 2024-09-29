import base64
import random
import boto3
import json
from image_data import metadata
from predictor import predict, load_trained_network
from io import BytesIO
from PIL import Image

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
    n = min(max(1,n), 100)

    # Select n random entries from the metadata
    random_metadata = random.sample(metadata, n)

    # Initialize S3 client
    s3 = boto3.client('s3', region_name=REGION)

    # Generate pre-signed URLs based on the selected metadata
    image_urls = [ generate_s3_presigned_url(s3, S3_BUCKET, entry) for entry in random_metadata ]

    return image_urls


def load_mnist_image(image_key):
    # Create an S3 client
    s3 = boto3.client('s3')

    # Get the image from S3
    response = s3.get_object(Bucket=S3_BUCKET, Key=image_key)
    image_data = response['Body'].read()

    # Load the image using PIL
    image = Image.open(BytesIO(image_data))

    # Convert the image to a flattened list of pixel values
    pixel_values = list(image.getdata())

    # Normalize the pixel values to be between 0 and 1
    normalized_values = [pixel / 255.0 for pixel in pixel_values]

    return normalized_values


def object_name_from_filename(filename):
    # Extract serial num from filename
    serial_num = int(filename.split('-')[0])

    valueA = serial_num // 1000
    valueB = (serial_num % 1000) // 100

    # Create the S3 object path
    return f"{valueA}/{valueB}/{filename}"



def predict_value(filename):
    object_name = object_name_from_filename(filename)
    filedata = load_mnist_image(object_name)
    network = load_trained_network()
    predicted_value = predict(network, filedata)
    return predicted_value


# Lambda handler function
def lambda_handler(event, context):
    http_method = event.get('requestContext', {}).get('http', {}).get('method', 'missing')
    if http_method == 'missing':
        return {
            "statusCode": 400,
            "body": json.dumps({'error': 'malformed http request'})
        }
    path = event.get('requestContext', {}).get('http', {}).get('path', 'missing')
    if http_method == 'missing':
        return {
            "statusCode": 400,
            "body": json.dumps({'error': 'malformed path'})
        }

    print(f'method: {http_method} and path: {path}')
    if http_method == 'GET' and path.endswith('/urls'):
        return handle_get_urls(event)
    elif http_method == 'GET' and path.endswith('/ui'):
        return handle_get_ui(event)
    elif http_method == 'POST' and '/prediction' in path:
        return handle_post_prediction(event)
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({'error': f'unrecognized method/path: "{http_method} {path}"'})
        }

# route handler functions
def handle_get_ui(event):
    print('handle_get_ui called')

    with open('app.html', 'r') as file:
        content = file.read()

    encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')

    print('handle_get_ui called: ', encoded_content)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html",
        },
        "body": encoded_content,
        "isBase64Encoded": True
    }

def handle_get_urls(event):
    n = int(event.get('queryStringParameters', {}).get('n', 100))  # Default to 100 if n is not provided
    data = get_random_image_urls(n)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(data)
    }

def handle_post_prediction(event):
    filename = event.get('pathParameters', {}).get('filename', '')
    if not filename:
        return {
            "statusCode": 400,
            "body": json.dumps({'error':'no filename specified'})
        }

    predicted_value = predict_value(filename)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({'predicted_value': predicted_value})
    }


# For testing locally
if __name__ == '__main__':
    random_image_urls = get_random_image_urls(n=10)
    for image in random_image_urls:
        print(image)
