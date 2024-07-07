import os
import cv2
import numpy as np
from PIL import Image
import shutil
import yaml

# Get the current working directory
current_directory = os.getcwd()

# Directory containing the original test set images and annotations
ORIGINAL_TEST_SET_DIR = os.path.join(current_directory, 'test_datasets/clean/valid/images')
ORIGINAL_ANNOTATIONS_DIR = os.path.join(current_directory, 'test_datasets/clean/valid/labels')

# Base directory to save the new test sets
BASE_OUTPUT_DIR = os.path.join(current_directory, 'test_datasets')
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Path to the original data.yaml file
DATA_YAML_PATH = os.path.join(current_directory, 'test_datasets/clean/data.yaml')

def add_gaussian_noise(image, mean=0, percentage=10.0):
    """
    Add Gaussian noise to an image.

    Parameters:
    image (numpy.ndarray): Input image.
    mean (float): Mean of the Gaussian noise.
    percentage (float): Percentage of noise to add.

    Returns:
    numpy.ndarray: Noisy image.
    """
    var = (percentage / 100.0) * 10000  # Increased variance for more noticeable noise
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    return noisy_image

def add_gaussian_blur(image, percentage=10.0):
    """
    Add Gaussian blur to an image.

    Parameters:
    image (numpy.ndarray): Input image.
    percentage (float): Percentage of blur to add.

    Returns:
    numpy.ndarray: Blurred image.
    """
    ksize = int((percentage / 100.0) * 50) + 2  # Increased kernel size for more noticeable blur
    ksize = (ksize if ksize % 2 == 1 else ksize + 1)  # Ensure kernel size is odd
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image

def modify_and_save_images_and_annotations(mode, percentages):
    """
    Modify and save images and annotations based on the given mode and percentages.

    Parameters:
    mode (str): Mode of modification ('noise' or 'blur').
    percentages (list of float): List of percentages for modification.
    """
    for percentage in percentages:
        subdir = f'{mode}_{percentage}_percent'
        save_dir = os.path.join(BASE_OUTPUT_DIR, subdir)
        images_save_dir = os.path.join(save_dir, 'images')
        annotations_save_dir = os.path.join(save_dir, 'labels')
        os.makedirs(images_save_dir, exist_ok=True)
        os.makedirs(annotations_save_dir, exist_ok=True)

        for filename in os.listdir(ORIGINAL_TEST_SET_DIR):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(ORIGINAL_TEST_SET_DIR, filename)
                image = Image.open(image_path).convert('RGB')
                image_np = np.array(image)

                if mode == 'noise':
                    modified_image = add_gaussian_noise(image_np, percentage=percentage)
                elif mode == 'blur':
                    modified_image = add_gaussian_blur(image_np, percentage=percentage)
                else:
                    print(f"Invalid mode: {mode}")
                    return

                modified_image_path = os.path.join(images_save_dir, filename)
                Image.fromarray(modified_image).save(modified_image_path)

                # Copy the corresponding annotation file
                annotation_filename = os.path.splitext(filename)[0] + '.txt'
                annotation_path = os.path.join(ORIGINAL_ANNOTATIONS_DIR, annotation_filename)
                if os.path.exists(annotation_path):
                    shutil.copy(annotation_path, annotations_save_dir)

        # Create and update the data.yaml file
        new_data_yaml_path = os.path.join(save_dir, 'data.yaml')
        data_yaml = {
            'test': '../test/images',
            'train': '../train/images',
            'val': '../images',
            'nc': 1,
            'names': ['SurfaceCrack']
        }
        with open(new_data_yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        print(f"Modified test set with {mode} at {percentage}% saved in {save_dir}.")

# User inputs
mode = input("Enter mode (noise/blur): ").strip().lower()
num_sets = int(input("Enter number of sets: ").strip())

percentages = []
for i in range(num_sets):
    percentage = float(input(f"Enter percentage for set {i+1}: ").strip())
    percentages.append(percentage)

# Modify and save images and annotations
modify_and_save_images_and_annotations(mode, percentages)
