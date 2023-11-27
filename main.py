import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import ttk

# import model
from ultralytics import YOLO
from cnn.cnn_yolov8 import cnn_yolov8
from classic.classic_clf import YOALAH

# mengkonfigurasi GUI
root = tk.Tk()
root.geometry("1000x600")
root.title("Vehicle Recognition")
root.config(bg="white")

file_path = ""

# Fungsi untuk memilih gambar dengan membuka file explorer
def choose_image():
    global file_path
    file_path = filedialog.askopenfilename(
        initialdir="./")
    image = Image.open(file_path)
    canvas.config(width=image.width, height=image.height)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")

# Menghapus citra dari canvas
def clear_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, image=canvas.image, anchor="nw")

# Memproses rekognisi
# Ada metode Classical dan CNN (YOLOv8)
def process():
    image = Image.open(file_path)
    
    method = method_combobox.get()
    if method == "Classical":
        image = classic(image)
        image = Image.fromarray(image)

    elif method == "CNN (YOLOv8)":
        image = cnn_yolov8(image)
        image = Image.fromarray(image)
    
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")

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

root.mainloop()
