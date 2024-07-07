import random
import shutil
import cv2
import numpy as np
from pathlib import Path
import yaml

# Get the current working directory
current_directory = Path.cwd()

def add_gaussian_noise(image, mean=0, percentage=20.0):
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

def add_gaussian_blur(image, percentage=20.0):
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

# Paths for the original and new dataset directories
original_dataset_path = current_directory / "training_cracks/DawgSurfaceCracks"
new_dataset_path = current_directory / "training_cracks/degraded_dataset"

# Create new dataset directory, removing existing one if necessary
if new_dataset_path.exists():
    shutil.rmtree(new_dataset_path)
shutil.copytree(original_dataset_path, new_dataset_path)

# Get list of all image files in the dataset
image_files = list(new_dataset_path.glob("**/*.jpg"))

# Randomly shuffle the images
random.shuffle(image_files)

# Split images into thirds for different processing
total_images = len(image_files)
blur_images = image_files[:total_images // 3]
noise_images = image_files[total_images // 3:2 * total_images // 3]
original_images = image_files[2 * total_images // 3:]

# Apply blur to the first third of images
for image_path in blur_images:
    image = cv2.imread(str(image_path))
    modified_image = add_gaussian_blur(image, 20.0)
    cv2.imwrite(str(image_path), modified_image)

# Apply noise to the second third of images
for image_path in noise_images:
    image = cv2.imread(str(image_path))
    modified_image = add_gaussian_noise(image, 0, 20.0)
    cv2.imwrite(str(image_path), modified_image)

print(f"Processed {len(blur_images)} images with blur and {len(noise_images)} images with noise.")

# Create the data.yaml file
data_yaml = {
    'train': '../train/images',
    'val': '../valid/images',
    'test': '../test/images',
    'nc': 1,
    'names': ['SurfaceCrack']
}

data_yaml_path = new_dataset_path / 'data.yaml'
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"data.yaml file created at {data_yaml_path}")
