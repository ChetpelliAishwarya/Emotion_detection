
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from facial_expression_model import FacialExpressionModel

# Initialize the main window
root = tk.Tk()
root.title("Emotion Detection")

# Load the model
try:
    model = FacialExpressionModel(r"C:\Users\chais\Desktop\Emotion_detection\model_.weights1.h5")
except Exception as e:
    print(f"Error loading model: {e}")

def load_image():
    global img_display
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    print(f"Loading image from {file_path}")
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Original image shape: {img.shape}")

    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, (1, 48, 48, 3)) / 255.0
    print(f"Processed image shape: {img.shape}")

    try:
        emotion = model.predict_emotion(img)
        print(f"Predicted emotion: {emotion}")
    except Exception as e:
        emotion = "Error"
        print(f"Error predicting emotion: {e}")
    
    # Display the image
    img_display = Image.fromarray(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB))
    img_display = img_display.resize((400, 400), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_display)
    panel.configure(image=img_tk)
    panel.image = img_tk
    label.configure(text=f"Emotion: {emotion}")

# Create GUI elements
panel = tk.Label(root)
panel.pack()

label = tk.Label(root, text="Emotion: ", font=("Arial", 14))
label.pack()

button = tk.Button(root, text="Load Image", command=load_image)
button.pack()

# Start the main loop
root.mainloop()
