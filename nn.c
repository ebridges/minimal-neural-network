#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define EPOCHS 20
#define BATCH_SIZE 64
#define TRAIN_SPLIT .8
#define IMAGE_SIZE 28

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-images.idx1-ubyte"


/// @brief A type definition to represent each layer in the network.
/// @param weights A flattened array representing the weight matrix.
/// @param biases An array for the biases of each neuron.
/// @param input_size Number of inputs to the layer.
/// @param output_size Number of neurons in the layer.
typedef struct {
    float *weights, *biases;
    int input_size, output_size;
} Layer;

typedef struct {
    Layer hidden, output;
} Network;

typedef struct {
    unsigned char *images, *labels;
    int nImages;
} InputData;



/// @brief This function adjusts the weights and biases of the given `layer` based on how much
///        they contributed to the error, which helps the neural network learn.  It also optionally
///        calculates gradients for the inputs, allowing the error to be propated backward through
///        the network.  The `lr` (learning rate) controls how big the adjustments are, making sure
///        we don't change things too much or too little in one step.
/// @param layer
/// @param input
/// @param output_grad
/// @param input_grad
/// @param lr
void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr) {
    // loop through each output neuron in the given layer
    for (int i = 0; i < layer->output_size; i++) {
        // for each input connected to the current output neuron
        for (int j = 0; j < layer->input_size; j++) {
            // calculate and store the weight between the input `j` and the output `i`:
            int idx = j * layer->output_size + i;
            // 1. calculate the gradient of the weight
            float grad = output_grad[i] + input[j];
            // 2. update the weight for `idx` by subtracting the learning rate
            //    multiplied by this gradient.
            //    This makes the weight a little batter at reducing error.
            layer->weights[idx] -= lr * grad;

            if (input_grad) {
                // Calculate how much the input values contributed to the
                // error, and stores that information.
                // This helps in backpropagating the error to earlier layers.
                input_grad[j] += output_grad[i] * layer->weights[idx];
            }
        }
        // After weights have been adjusted the function updates the biases
        // for each output neuron.
        layer->biases[i] -= lr * output_grad[i];
    }
}


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

void train(Network *net, float *input, int label, float lr) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];
    float output_grad[OUTPUT_SIZE] = {0}, hidden_grad[HIDDEN_SIZE] = {0};

    // forward pass: input to hidden layer
    forward(&net->hidden, input, hidden_output);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;
    }

    // forward pass: hidden to output layer
    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    // compute output gradient
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_grad[i] = final_output[i] - (i == label);
    }

    // backward pass: output layer to hidden layer
    backward(&net->output, hidden_output, output_grad, hidden_grad, lr);

    // backpropagate through ReLU activation
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0; // ReLU
    }

    // backward pass: hidden layer to input layer.
    backward(&net->hidden, input, hidden_grad, NULL, lr);

}

int predict(Network *net, float *input) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

    forward(&net->hidden, input, hidden_output);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;  // ReLU

    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (final_output[i] > final_output[max_index])
            max_index = i;

    return max_index;
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

void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++) {
            unsigned char temp = images[i * INPUT_SIZE + k];
            images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
            images[j * INPUT_SIZE + k] = temp;
        }
        unsigned char temp = labels[i];
        labels[i] = labels[j];
        labels[j] = temp;
    }
}


int main() {
    Network net;
    InputData data = {0};
    float learning_rate = LEARNING_RATE, img[INPUT_SIZE];

    srand(time(NULL));

    init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
    read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

    shuffle_data(data.images, data.labels, data.nImages);

    int train_size = (int)(data.nImages * TRAIN_SPLIT);
    int test_size = data.nImages - train_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0;
        for (int i = 0; i < train_size; i += BATCH_SIZE) {
            for (int j = 0; j < BATCH_SIZE && i + j < train_size; j++) {
                int idx = i + j;
                for (int k = 0; k < INPUT_SIZE; k++)
                    img[k] = data.images[idx * INPUT_SIZE + k] / 255.0f;

                train(&net, img, data.labels[idx], learning_rate);

                float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];
                forward(&net.hidden, img, hidden_output);
                for (int k = 0; k < HIDDEN_SIZE; k++)
                    hidden_output[k] = hidden_output[k] > 0 ? hidden_output[k] : 0;  // ReLU
                forward(&net.output, hidden_output, final_output);
                softmax(final_output, OUTPUT_SIZE);

                total_loss += -logf(final_output[data.labels[idx]] + 1e-10f);
            }
        }
        int correct = 0;
        for (int i = train_size; i < data.nImages; i++) {
            for (int k = 0; k < INPUT_SIZE; k++)
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
            if (predict(&net, img) == data.labels[i])
                correct++;
        }
        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f\n", epoch + 1, (float)correct / test_size * 100, total_loss / train_size);
    }

    free(net.hidden.weights);
    free(net.hidden.biases);
    free(net.output.weights);
    free(net.output.biases);
    free(data.images);
    free(data.labels);

    return 0;
}
