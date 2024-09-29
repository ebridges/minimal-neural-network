from math import exp, sqrt
from json import load
from random import random
from typing import List

INPUT_SIZE=784 # 28x28 pixel grid
HIDDEN_SIZE=256
OUTPUT_SIZE=10
TRAINED_MODEL_FILE='trained-model.json'


class Layer:
    def kaiming_he(input_sz : float, total_sz : float) -> List[float]:
        scale = sqrt(float(2) / input_sz)
        return [random() * float(2) * scale for _ in range(total_sz)]

    def __init__(self, input_size : int, output_size : int):
        self.weights : List[float] = Layer.kaiming_he(input_size, input_size * output_size)
        self.biases : List[float] = [float(0)] * output_size
        self.input_size : int = input_size
        self.output_size : int = output_size


    def new_from_model(self, weights, biases, input_sz, output_sz):
        self.weights = weights
        self.biases = biases
        self.input_size = input_sz
        self.output_size = output_sz

class Network:
    __instance = None

    @staticmethod
    def trained_instance(model_file):
        """ Static access method. """
        if Network.__instance == None:
            Network(model_file)

        return Network.__instance

    def __init__(self,model_file):
        """ Virtually private constructor. """
        if Network.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            with open(model_file, 'r') as file:
                model = load(file)
            h = model['hidden']
            self.hidden = Layer.new_from_model(h.weights, h.biases, h.input_size, h.output_size)
            o = model['output']
            self.output = Layer.new_from_model(o.weights, o.biases, o.input_size, o.output_size)
            Network.__instance = self


    def __init__(self, input_sz : int, hidden_sz : int, output_sz : int):
        self.hidden : Layer = Layer(input_sz, hidden_sz)
        self.output: Layer = Layer(hidden_sz, output_sz)


def load_trained_network():
    return Network.trained_instance(TRAINED_MODEL_FILE)


def softmax(input : List[float]):
    sum = float()
    mx = max(input)
    for i in range(0, len(input)):
        input[i] = exp(input[i] - mx)
        sum += input[i]
    for i in range(0, len(input)):
        input[i] /= sum


def forward(layer: Layer, input: List[float], output: List[float]):
    for i in range(layer.output_size):
        output[i] = layer.biases[i]
        for j in range(layer.input_size):
            output[i] += input[j] * layer.weights[j * layer.output_size + i]


def predict(network : Network, image : List[float]) -> int:
    hidden_output : List[float] = [float(0)] * HIDDEN_SIZE
    final_output : List[float] = [float(0)] * OUTPUT_SIZE

    forward(network.hidden, image, hidden_output)
    for i in range(HIDDEN_SIZE):
        hidden_output[i] = max(hidden_output[i], 0)

    forward(network.output, hidden_output, final_output)
    softmax(final_output)

    max_idx = 0
    for i in range(OUTPUT_SIZE):
        if final_output[i] > final_output[max_idx]:
            max_idx = i

    return max_idx





















