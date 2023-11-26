import cv2
from skimage.feature import hog
from skimage import exposure
import numpy as np

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Histogram of Oriented Gradients (HOG)
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    # Return the HOG features
    return features

# Load an example image (replace this with your image loading logic)
image_path = './car.png'
image = cv2.imread(image_path)

# Extract HOG features
hog_features = extract_hog_features(image)

# Print the shape of the extracted features
print("HOG Features Shape:", hog_features.shape)
