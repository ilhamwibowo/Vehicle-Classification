import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter
from tkinter import ttk
from ultralytics.utils.plotting import plot_labels

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

from classic.classic_clf import YOALAH

root = tk.Tk()
root.geometry("1000x600")
root.title("Vehicle Recognition")
root.config(bg="white")

file_path = ""

def choose_image():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="./")
    image = Image.open(file_path)
    width, height = int(image.width / 2), int(image.height / 2)
    image = image.resize((width, height), Image.LANCZOS)
    canvas.config(width=image.width, height=image.height)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")
    
def clear_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, image=canvas.image, anchor="nw")

def process():
    image_original = Image.open(file_path)
    width, height = int(image_original.width / 2), int(image_original.height / 2)
    image = image_original.resize((width, height), Image.LANCZOS)
    
    method = method_combobox.get()
    if method == "Classical":
        image = classic(image)
        image = Image.fromarray(image)

    elif method == "CNN (YOLOv8)":
        results = cnn_yolov8(image)

        image = plot_yolov8_bboxes(np.asarray(image), results[0].boxes.cpu(), labels=results[0].names, score=True, conf=None)
        image = Image.fromarray(image)
    
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")

def cnn_yolov8(image):
    model = YOLO("yolov8x.pt")
    detections = model.predict(source=image, save=False)

    return detections

def classic(image):
    classifier = YOALAH('classic/trained_model.pkl')
    return classifier.predict(image)

left_frame = tk.Frame(root, width=200, height=600, bg="white")
left_frame.pack(side="left", fill="y")

canvas = tk.Canvas(root, width=750, height=600)
canvas.pack()

image_button = tk.Button(left_frame, text="Choose Image",
                         command=choose_image, bg="white")
image_button.pack(pady=15)

clear_button = tk.Button(left_frame, text="Clear",
                         command=clear_canvas, bg="#FF9797")
clear_button.pack(pady=10)


filter_label = tk.Label(left_frame, text="Select Method", bg="white")
filter_label.pack()
method_combobox = ttk.Combobox(left_frame, values=["Classical", "CNN (YOLOv8)"])
method_combobox.pack()

image_button = tk.Button(left_frame, text="Process",
                         command=process, bg="white")
image_button.pack(pady=15)

# ======== Menampilkan box hasil deteksi YOLOv8 ===========
# plot bounding box hasil deteksi YOLOv8
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

# plot satu box
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    x,y,w,h = box.xywh[0]
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    image = cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 1)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

root.mainloop()
