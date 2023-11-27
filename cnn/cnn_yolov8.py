from ultralytics import YOLO
import cv2
import numpy as np

# Melakukan deteksi dengan YOLOv8
def cnn_yolov8(image):
    # Load weights YOLO
    model = YOLO("yolov8m.pt")

    # Melakukan prediksi dengan YOLO
    detections = model.predict(source=image, save=False)

    # Menggambar bounding box hasil prediksi ke citra
    image = plot_yolov8_bboxes(np.asarray(image), detections[0].boxes.cpu(), labels=detections[0].names, score=True, conf=None)

    return image

# ======== Memberikan tanda box hasil deteksi YOLOv8 ===========
# Plot bounding box hasil deteksi YOLOv8
def plot_yolov8_bboxes(image, boxes, labels=[], score=True, conf=None):
    # warna box
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]
    
    # plot tiap box
    for box in boxes:
        # menambahkan nilai score (jika score = True)
        if score :
            label = labels[int(box.cls)] + " " + str(round(100 * float(box.conf),1)) + "%"
        else :
            label = labels[int(box.cls)]

        # hanya menampilkan deteksi di atas conf (jika conf != None)
        if conf:
            if box[-2] > conf:
                color = colors[int(box.cls)]
                box_label(image, box, label, color)
        else:
            color = colors[int(box.cls)]
        box_label(image, box, label, color)

    return image

# Plot satu box saja
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(0, 255, 0)):
    # Ekstrak nilai x, y, w, dan h
    x,y,w,h = box.xywh[0]
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    # Menambahkan kotak ke citra
    image = cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 1)

    # Menambahkan text kelas ke citra
    cv2.putText(image, label, (x, y+h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color, 2)
