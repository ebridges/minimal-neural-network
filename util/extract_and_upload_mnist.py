#!/usr/bin/env python3

import os
import struct
import numpy as np
from PIL import Image
import boto3
from botocore.exceptions import NoCredentialsError

def extract_mnist_images(image_file, label_file, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    serial_num = 0

    with open(image_file, 'rb') as img_f, open(label_file, 'rb') as lbl_f:
        # Read header for images
        img_magic, img_num, img_rows, img_cols = struct.unpack('>IIII', img_f.read(16))

        if img_magic != 2051:
            raise ValueError("Image file not in expected MNIST format. Expected magic number 2051.")

        # Read header for labels
        lbl_magic, lbl_num = struct.unpack('>II', lbl_f.read(8))

        if lbl_magic != 2049:
            raise ValueError("Label file not in expected MNIST format. Expected magic number 2049.")

        assert img_num == lbl_num, "Number of images and labels must match."

        for i in range(img_num):
            # Read image data
            img_data = img_f.read(img_rows * img_cols)
            buffer = np.frombuffer(img_data, dtype=np.uint8)
            image = buffer.reshape((img_rows, img_cols))

            # Read label data
            label = struct.unpack('>B', lbl_f.read(1))[0]

            # Generate a UUID for the image
            img_filename = f"{serial_num:05d}-{label}"
            serial_num += 1
            img_path = os.path.join(output_dir, f'{img_filename}.png')

            # Save the image as PNG
            img = Image.fromarray(image)
            img = img.convert("L")  # Convert to grayscale
            img.save(img_path)
            print(f"Saved: {img_path}")

            # Save the image as byte data
            normalized_values = (buffer.astype(np.float32) / 255.0).tobytes()  # Convert to 32-bit float and normalize
            raw_path = os.path.join(output_dir, f'{img_filename}.raw')
            with open(raw_path, 'wb') as raw_f:
                raw_f.write(normalized_values)



def object_name_from_filename(filename):
    # Extract serial num from filename
    serial_num = int(filename.split('-')[0])

    valueA = serial_num // 1000
    valueB = (serial_num % 1000) // 100

    # Create the S3 object path
    return f"{valueA}/{valueB}/{filename}"


def upload_images_to_s3(output_dir, bucket_name):
    s3_client = boto3.client('s3')

    for img_filename in os.listdir(output_dir):
        if img_filename.endswith('.png') or img_filename.endswith('.raw') :
            img_path = os.path.join(output_dir, img_filename)

            s3_object_name = object_name_from_filename(img_filename)

            try:
                s3_client.upload_file(img_path, bucket_name, s3_object_name)
                print(f"Uploaded: {s3_object_name} to bucket: {bucket_name}")
            except NoCredentialsError:
                print("Credentials not available")


def main(image_file, label_file, output_dir, bucket_name):
    extract_mnist_images(image_file, label_file, output_dir)
    upload_images_to_s3(output_dir, bucket_name)


def list_objects(output_dir):
    for img_filename in os.listdir(output_dir):
        if img_filename.endswith('.png'):
            print(object_name_from_filename(img_filename))

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Extract MNIST images and upload to S3.")
    # parser.add_argument("image_file", help="Path to the MNIST images file.")
    # parser.add_argument("label_file", help="Path to the MNIST labels file.")
    # parser.add_argument("output_dir", help="Output directory to save PNG images.")
    # parser.add_argument("bucket_name", help="S3 bucket name for uploading images.")

    # args = parser.parse_args()

    # main(args.image_file, args.label_file, args.output_dir, args.bucket_name)
    list_objects('img')
