import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import re
from skimage import color
from . import preprocess
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

class YOALAH:
    def __init__(self, pre_trained_model=None):
        self.clf = svm.SVC(kernel='linear')

        if pre_trained_model:
            self.clf = load(pre_trained_model)
        else:
            self.train_model()

    def load_dataset(self, folder_path):
        data = []
        labels = []
        
        pattern = re.compile(r'(CAR|BUS|TRUCK|AMBULANCE)', flags=re.IGNORECASE)
        target_size = (100, 100) 

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image = cv2.imread(os.path.join(folder_path, filename))
                image = cv2.resize(image, target_size)

                match = pattern.search(filename)
                if match:
                    label = match.group().upper()
                else:
                    label = 'UNKNOWN'

                data.append(image)
                labels.append(label)

        data = np.array(data)
        labels = np.array(labels)
        
        # Flatten
        n_samples, height, width, channels = data.shape
        data = data.reshape((n_samples, -1))
        
        return data, labels

    def train_model(self):
        folder_path = 'archive/training_image'
        data, labels = self.load_dataset(folder_path)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    def save_model(self, file_name='trained_model.pkl'):
        joblib.dump(self.clf, file_name)

    def predict_classes(self, cropped_images):
        # Resize 
        resized_images = [cv2.resize(img, (100, 100)) for img in cropped_images]

        # Flatten 
        flattened_images = np.array([img.flatten() for img in resized_images])

        # Predict
        predictions = self.clf.predict(flattened_images)
        return predictions

    def bounded_image_with_prediction(self, bounded_image, bounding_boxes, predictions):
        for bbox, prediction in zip(bounding_boxes, predictions):
            x1, y1, x2, y2 = bbox

            bounded_region = bounded_image[y1:y2, x1:x2]

            class_label = f"{prediction}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(class_label, font, font_scale, thickness)[0]

            background_color = (0, 0, 255)
            text_origin = (10, 20) 
            text_end = (text_origin[0] + text_size[0] + 2, text_origin[1] - text_size[1] - 2)
            cv2.rectangle(bounded_region, text_origin, text_end, background_color, -1)

            cv2.putText(bounded_region, class_label, text_origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # plt.imshow(cv2.cvtColor(bounded_image, cv2.COLOR_BGR2RGB))
        # plt.title('Original Image with Bounding Boxes and Predictions')
        # plt.axis('off')
        # plt.show()

    def predict(self, image):
        original_image = np.array(image)

        if original_image.shape[-1] == 3: 
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        elif original_image.shape[-1] == 4:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)

        segmented_image = preprocess.segment(original_image) # segment
        
        # terus kotak-kotakin
        coordinates, bounded_image = preprocess.bounding_box(original_image, segmented_image)

        # ambil gambar tiap kotak
        cropped_regions = preprocess.extract_bounding_boxes(original_image, coordinates)

        # predict tiap kotak
        predicted_classes = self.predict_classes(cropped_regions)
        print(predicted_classes)

        # display kotak-kotak
        self.bounded_image_with_prediction(bounded_image, coordinates, predicted_classes)

        return cv2.cvtColor(bounded_image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('result.jpg', bounded_image)

# if __name__ == "__main__" :
#     # folder_path = 'archive/training_image'
#     # data, labels = load_dataset(folder_path)
#     # print("Data shape:", data.shape)
#     # print("Labels shape:", labels.shape)

#     # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#     # clf = svm.SVC(kernel='linear')

#     # # Train 
#     # clf.fit(X_train, y_train)

#     # # test
#     # y_pred = clf.predict(X_test)

#     # accuracy = accuracy_score(y_test, y_pred)
#     # print("Accuracy:", accuracy)

#     # dump(clf, 'filename.joblib')

#     # Train the model
#     # obj_detector = ClassicObjectDetection()
#     # obj_detector.save_model()

#     obj_detector = ClassicObjectDetection('trained_model.pkl')

#     out = obj_detector.predict("archive/training_image/8Ambulance.jpg")
