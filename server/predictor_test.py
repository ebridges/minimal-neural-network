import unittest
from math import exp, sqrt
from predictor import Layer, Network, forward, softmax, predict, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

class TestNeuralNetwork(unittest.TestCase):

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
        network = Network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

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
