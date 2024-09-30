import unittest
import random
from json import dumps
from unittest.mock import mock_open, patch
from math import exp, sqrt
from predictor import Layer, Network, forward, softmax, predict, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Sample data representing a trained model in JSON format
        self.model_data = {
            'hidden': {
                'input_size': INPUT_SIZE,
                'output_size': HIDDEN_SIZE,
                'weights': [random() for _ in range(INPUT_SIZE * HIDDEN_SIZE)],
                'biases': [random() for _ in range(HIDDEN_SIZE)]
            },
            'output': {
                'input_size': HIDDEN_SIZE,
                'output_size': OUTPUT_SIZE,
                'weights': [random() for _ in range(HIDDEN_SIZE * OUTPUT_SIZE)],
                'biases': [random() for _ in range(OUTPUT_SIZE)]
            }
        }

        # Convert to JSON string
        self.model_json = dumps(self.model_data)

    @patch('builtins.open', new_callable=mock_open, read_data=dumps({}))
    @patch('json.load')
    def test_init_from_model(self, mock_json_load, mock_file):
        """ Test the init_from_model constructor of the Network class """
        # Set the mock return value for json.load to the model data
        mock_json_load.return_value = self.model_data

        # Call the method
        network = Network.init_from_model('dummy_model_file.json')

        # Check that the hidden layer is properly initialized
        self.assertIsNotNone(network.hidden)
        self.assertEqual(network.hidden.input_size, self.model_data['hidden']['input_size'])
        self.assertEqual(network.hidden.output_size, self.model_data['hidden']['output_size'])
        self.assertEqual(network.hidden.weights, self.model_data['hidden']['weights'])
        self.assertEqual(network.hidden.biases, self.model_data['hidden']['biases'])

        # Check that the output layer is properly initialized
        self.assertIsNotNone(network.output)
        self.assertEqual(network.output.input_size, self.model_data['output']['input_size'])
        self.assertEqual(network.output.output_size, self.model_data['output']['output_size'])
        self.assertEqual(network.output.weights, self.model_data['output']['weights'])
        self.assertEqual(network.output.biases, self.model_data['output']['biases'])

        # Ensure the file was opened and read
        mock_file.assert_called_once_with('dummy_model_file.json', 'r')
        mock_json_load.assert_called_once()

    def tearDown(self):
        # Clean up Network singleton state after each test
        Network._Network__instance = None  # Reset the singleton instance



    def test_layer_initialization(self):
        input_size = 3
        output_size = 2

        # Initialize a Layer instance
        layer = Layer(input_size, output_size)

        # Test weights length: should be input_size * output_size
        expected_weight_length = input_size * output_size
        self.assertEqual(len(layer.weights), expected_weight_length)

        # Test weights values: ensure they are within the expected range (from 0 to 2 * scale)
        scale = sqrt(2.0 / input_size)
        for weight in layer.weights:
            self.assertTrue(0 <= weight < 2 * scale)

        # Test biases: should be a list of zeros with length equal to output_size
        self.assertEqual(len(layer.biases), output_size)
        self.assertTrue(all(bias == 0.0 for bias in layer.biases))

        # Test input_size and output_size are correctly set
        self.assertEqual(layer.input_size, input_size)
        self.assertEqual(layer.output_size, output_size)


    def test_softmax(self):
        # Test that softmax output is a valid probability distribution
        input_data = [2.0, 1.0, 0.1]
        expected_output = [exp(2.0)/sum([exp(2.0), exp(1.0), exp(0.1)]),
                           exp(1.0)/sum([exp(2.0), exp(1.0), exp(0.1)]),
                           exp(0.1)/sum([exp(2.0), exp(1.0), exp(0.1)])]
        softmax(input_data)
        self.assertAlmostEqual(sum(input_data), 1.0, places=6)  # Ensure probabilities sum to 1
        for i in range(len(input_data)):
            self.assertAlmostEqual(input_data[i], expected_output[i], places=6)


    def test_forward(self):
        # Simple test for forward pass with fixed weights and biases
        layer = Layer(3, 2)  # 3 input neurons, 2 output neurons
        # Set specific weights and biases
        layer.weights = [0.2, 0.4, 0.6, 0.1, 0.3, 0.5]
        layer.biases = [0.1, -0.1]

        input_data = [1.0, 2.0, 3.0]  # Test input
        output = [0.0, 0.0]  # Placeholder for output

        forward(layer, input_data, output)

        # Expected output based on the calculations:
        # output[0] = 0.2*1.0 + 0.6*2.0 + 0.3*3.0 + 0.1 = 2.4
        # output[1] = 0.4*1.0 + 0.1*2.0 + 0.5*3.0 - 0.1 = 2.0
        self.assertAlmostEqual(output[0], 2.4, places=6)
        self.assertAlmostEqual(output[1], 2.0, places=6)


    def test_predict(self):
        # Mock a simple network for testing
        network = Network.create_for_testing(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        # Set up predefined weights and biases for predict function
        network.hidden.weights = [0.2] * (INPUT_SIZE * HIDDEN_SIZE)
        network.hidden.biases = [0.1] * HIDDEN_SIZE
        network.output.weights = [0.3] * (HIDDEN_SIZE * OUTPUT_SIZE)
        network.output.biases = [0.2] * OUTPUT_SIZE

        # Test image (flattened)
        test_image = [0.5] * INPUT_SIZE

        # Call predict method
        predicted_label = predict(network, test_image)

        # The result should be an integer label between 0 and 9
        self.assertTrue(0 <= predicted_label < OUTPUT_SIZE)

        # Ensure the output is an integer
        self.assertIsInstance(predicted_label, int)


# Run the tests
if __name__ == '__main__':
    unittest.main()
