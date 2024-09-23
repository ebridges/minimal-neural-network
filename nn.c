#include <stdio.h>
#include <stdlib.h>

int IMAGE_SIZE = 28;

/// @brief A type definition to represent each layer in the network.
/// @param weights A flattened array representing the weight matrix.
/// @param biases An array for the biases of each neuron.
/// @param input_size Number of inputs to the layer.
/// @param output_size Number of neurons in the layer.
typedef struct {
    float *weights, *biases;
    int input_size, output_size;
} Layer;

/*
Apply the ReLU activation function in the hidden layers:

for (int i = 0; i < HIDDEN_SIZE; i++) {
    hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;
}
*/


/// @brief Compute probabilities by applying the softmax function to the output layer.
/// @param input
/// @param size
void softmax(float *input, int size) {
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i=0; i < size; i++) {
        input[i] /= sum;
    }
}


/// @brief Compute the output of an NN layer given an input.
/// @param layer
/// @param input
/// @param output
void forward(Layer *layer, float *input, float *output) {
    for (int i=0; i < layer->output_size; i++) {
        output[i] = layer->biases[i];
        for (int j=0; j < layer->input_size; j++) {
            output[i] += input[j] * layer->weights[j * layer->output_size + i];
        }
    }
}


/// @brief Initializes the weights & biases for a given layer.  Weights are set using Kaiming He Initialization.
/// @param layer
/// @param in_size
/// @param out_size
void init_layer(Layer *layer, int in_size, int out_size) {
    int n = in_size * out_size;

    layer->input_size = in_size;
    layer->output_size = out_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc(out_size, sizeof(float)); // init to 0

    float scale = sqrtf(2.0f / in_size);
    for (int i=0; i<n; i++) {
        // given limit of sqrt(2/in_size) the weight is set to a
        // value from the normal distribution `N(-limit, limit)`
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
}


/// @brief Allocate memory and read images from an IDX file into given pointer `images`.
/// @param filename
/// @param images
/// @param nImages
void read_mnist_images(const char * filename, unsigned char **images, int *nImages) {
    FILE *file = fopen(filename, "rb");
    if (!file)
        exit(1);

    int temp, rows, cols;
    fread(&temp, sizeof(int), 1, file);

    fread(nImages, sizeof(int), 1, file);
    *nImages = __builtin_bswap32(*nImages);

    fread(&rows, sizeof(int), 1, file);
    rows = __builtin_bswap32(rows);

    fread(&cols, sizeof(int), 1, file);
    cols = __builtin_bswap32(cols);

    *images = malloc((*nImages) * IMAGE_SIZE * IMAGE_SIZE);
    if (!images)
        exit(1);

    fread(*images, sizeof(unsigned char), (*nImages) * IMAGE_SIZE * IMAGE_SIZE, file);
    fclose(file);
}


/// @brief Allocate memory and read labels from an IDX file into the given pointer `labels`.
/// @param filename
/// @param labels
/// @param nLabels
void read_mnist_labels(const char* filename, unsigned char **labels, int *nLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file)
        exit(1);

    int temp;
    fread(&temp, sizeof(int), 1, file);

    fread(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    *labels = malloc(*nLabels);
    if (!labels)
        exit(1);

    fread(*labels, sizeof(unsigned char), *nLabels, file);
    fclose(file);
}

