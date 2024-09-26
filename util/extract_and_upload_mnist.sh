#!/bin/bash

output_dir='img'
bucket='com.eqbridges.mnist-img-archive'

./extract_and_upload_mnist.py \
    data/train-images.idx3-ubyte \
    data/train-labels.idx1-ubyte \
    ${output_dir} \
    ${bucket}
