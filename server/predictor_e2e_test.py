import unittest
import boto3
from io import BytesIO
from PIL import Image

from predictor import predict, Network

class TestNetworkIntegration(unittest.TestCase):
    def setUp(self):
        self.network = Network.init_from_model('trained-model.json')
        self.s3_bucket = 'com.eqbridges.mnist-archive'
        self.s3_key = '6/3/06312-4.png'


    def load_image_from_s3(self, bucket, key):
        """Load an image from S3 and return it as a flattened list."""
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()

        # Open the image using PIL
        image = Image.open(BytesIO(image_data)).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image_data = list(image.getdata())  # Get pixel data
        image_data = [(255 - pixel) // 255 for pixel in image_data]  # Invert colors and normalize

        return image_data

    def test_predict_raw_image_from_s3(self):
        """Test prediction with a raw image loaded from S3."""
        number_4_image = self.load_image_from_s3(self.s3_bucket, self.s3_key)
        predicted_value = predict(self.network, number_4_image)
        print(predicted_value)  # For debugging purposes
        self.assertEqual(predicted_value, 4, "The predicted value should be 4.")


# Running the tests
if __name__ == '__main__':
    unittest.main()
