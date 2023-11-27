
import cv2
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from skimage import color, filters
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_opening, rectangle, remove_small_objects
import numpy as np
import scipy.ndimage as ndi
from skimage.feature import hog
import networkx as nx

# ini return image yang udah dikotakin sama koordinat koordinat kotak
def bounding_box(original_image, segmented_image):
    
    gray_segmented = color.rgb2gray(segmented_image)
    thresh = (gray_segmented * 255).astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # kotak-kotakin boy
    result = original_image.copy()
    bounding_boxes = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        bounding_boxes.append((x, y, x + w, y + h)) 
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image with Red Bounding Boxes')
    # plt.axis('off')
    # plt.show()

    return bounding_boxes, result

# ini return array yang isinya --supposedly-- kendaraan kendaraan
def extract_bounding_boxes(image, bounding_boxes):
    cropped_images = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = box 
        cropped = image[y1:y2, x1:x2]
        cropped_images.append(cropped)

    return cropped_images

# ini buat ngetes ajah
def display_cropped_images(cropped_images):
    for idx, cropped in enumerate(cropped_images):
        plt.figure(figsize=(4, 4))
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.title(f"Cropped Image {idx + 1}")
        plt.axis('off')
        plt.show()

# fungsi melakukan segmentasi, masih buruk hasilnya
def segment(image):
    # x = plt.imread(image_path)
    image = np.array(image)
    if image.shape[2] == 4:
        image = color.rgba2rgb(x)

    x_gray = color.rgb2gray(image)
    edges = canny(x_gray)

    # structuring element-nya garis
    line_length = max(5, max(image.shape) // 35)
    
    horizontal_line = rectangle(1, line_length)

    # di-close, fill, trus open, trus buang noise noise
    closed_edges = binary_closing(edges, horizontal_line)
    fill_coins = ndi.binary_fill_holes(closed_edges)
    opened_edges = binary_opening(fill_coins, horizontal_line) 
    final_mask = remove_small_objects(opened_edges, min_size=1500)

    # Output Image
    red_processed = image[:, :, 0] * final_mask.astype(np.uint8)
    green_processed = image[:, :, 1] * final_mask.astype(np.uint8)
    blue_processed = image[:, :, 2] * final_mask.astype(np.uint8)
    output_image = np.stack([red_processed, green_processed, blue_processed], axis=-1)

    # Plotting
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # axes[0, 0].imshow(x)

    # axes[0, 0].set_title('Original Image')
    # axes[0, 0].axis('off')

    # axes[0, 1].imshow(closed_edges, cmap=plt.cm.gray)
    # axes[0, 1].set_title('Closing')
    # axes[0, 1].axis('off')

    # axes[0, 2].imshow(fill_coins, cmap=plt.cm.gray)
    # axes[0, 2].set_title('Filled Holes')
    # axes[0, 2].axis('off')

    # axes[1, 0].imshow(opened_edges, cmap=plt.cm.gray)
    # axes[1, 0].set_title('Opening')
    # axes[1, 0].axis('off')

    # axes[1, 1].imshow(final_mask, cmap=plt.cm.gray)
    # axes[1, 1].set_title('Mask')
    # axes[1, 1].axis('off')

    # axes[1, 2].imshow(output_image)
    # axes[1, 2].set_title('Output Image')
    # axes[1, 2].axis('off')

    # plt.tight_layout()
    # plt.show()

    return output_image


# feature extraction here
# def some feature extraction
def extract_hog_features(image):
    print(type(image))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_sift_features(image):
    print(type(image))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    min_response = 0.03  # Adjust the threshold as needed
    for i in range(5):
        print(keypoints[i].response)
    keypoints = [kp for kp in keypoints if kp.response > min_response]
    descriptors = sift.compute(gray, keypoints)[1]
    
    distances = np.linalg.norm(descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :], axis=-1)

    G = nx.Graph()
    for i in range(len(keypoints)):
        for j in range(i + 1, len(keypoints)):
            G.add_edge(i, j, weight=distances[i, j])
    
    mst_edges = nx.minimum_spanning_edges(G, algorithm='prim', data=False)

    mst_graph = nx.Graph()
    mst_graph.add_edges_from(mst_edges)

    selected_keypoints = []
    selected_descriptors = []
    for edge in mst_edges:
        selected_keypoints.extend([keypoints[edge[0]], keypoints[edge[1]]])
        selected_descriptors.extend([descriptors[edge[0]], descriptors[edge[1]]])
    
        if len(selected_keypoints) >= 20:
            break
    
    selected_keypoints = np.array(selected_keypoints)
    selected_descriptors = np.array(selected_descriptors)

    features = selected_descriptors.flatten()

    return features