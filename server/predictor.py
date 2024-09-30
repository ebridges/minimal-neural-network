from math import exp, sqrt
import json
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

    def __init__(self, input_size : int, output_size : int, weights: List[float] = None, biases: List[float] = None):
        if weights:
            self.weights = weights
        else:
            self.weights : List[float] = Layer.kaiming_he(input_size, input_size * output_size)
        if biases:
            self.biases = biases
        else:
            self.biases : List[float] = [float(0)] * output_size
        self.input_size : int = input_size
        self.output_size : int = output_size


class Network:
    __instance = None

    def __init__(self):
        if Network.__instance != None:
            raise Exception("This class is a singleton!")
        self.hidden = None
        self.output = None
        pass

    @classmethod
    def init_from_model(cls, model_file):
        if cls.__instance == None:
            cls.__instance = cls()
            with open(model_file, 'r') as file:
                model = json.load(file)
                h = model['hidden']
                cls.__instance.hidden = Layer(h['input_size'], h['output_size'], h['weights'], h['biases'])
                o = model['output']
                cls.__instance.output = Layer(o['input_size'], o['output_size'], o['weights'], o['biases'])
        return cls.__instance


    @classmethod
    def create_for_testing(cls, input_sz : int, hidden_sz : int, output_sz : int):
        cls.__instance = cls()
        cls.__instance.hidden = Layer(input_sz, hidden_sz)
        cls.__instance.output = Layer(hidden_sz, output_sz)
        return cls.__instance


def load_trained_network():
    return Network.init_from_model(TRAINED_MODEL_FILE)


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
