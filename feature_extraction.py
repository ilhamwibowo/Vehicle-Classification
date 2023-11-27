import cv2
from skimage.feature import hog
from skimage import exposure
import numpy as np

# Function to extract HOG features from an image
def extract_hog_features(image, labels):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features, labels

def extract_sift_features(image, labels):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    features = descriptors.flatten()

    return features, labels
