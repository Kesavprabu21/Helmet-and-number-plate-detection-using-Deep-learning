import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
import pytesseract
import os
import csv

# Initialize OpenCV window
cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)

# Load YOLOv3 model for object detection (persons and vehicles)
net = cv2.dnn.readNet("D:/projects/try/yolov3-custom_7000.weights", "D:/projects/try/yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the helmet detection model
helmet_model = load_model("D:/projects/try/helmet-nonhelmet_cnn (1).h5")

# Initialize Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Path to your Tesseract executable

# Specify the directory to save number plate images
output_dir = "D:/projects/try/numplate"
os.makedirs(output_dir, exist_ok=True)

# Initialize counts for persons with and without helmets
persons_with_helmet = 0
persons_without_helmet = 0

# Initialize a list to store detected persons and their helmet status
persons = []

# File path to save the CSV file
csv_file_path = "license_plate_sequences.csv"

# Initialize a list to store unique number plate character sequences along with image names
unique_plate_data = []

# Get YOLOv3 output layer names for persons and license plates
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the helmet_or_nohelmet function here
def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        prediction = helmet_model.predict(helmet_roi)
        if prediction is not None and len(prediction) > 0:
            return int(prediction[0][0])
        else:
            return 0  # Default to helmet (0) if prediction is None or empty
    except:
        return 0  # Default to helmet (0) in case of any exception

# Create a function to handle the "Start Processing" button click
def start_processing():
    global persons_with_helmet, persons_without_helmet, persons, unique_plate_data

    video_file_path = video_file_entry.get()
    if not video_file_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    # Open the video file for reading
    cap = cv2.VideoCapture(video_file_path)

    # Set up video writer for output
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter('output.avi', fourcc, 30, (888, 500))

    # Process video frames
    while True:
        ret, img = cap.read()

        if not ret:
            break  # End of video

        img = imutils.resize(img, height=500)
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        confidences = []
        boxes = []
        classIds = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIds.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Initialize variables to track persons in the current frame
        persons_in_frame = []

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                color = [int(c) for c in (0, 255, 0)] if classIds[i] == 0 else [int(c) for c in (0, 0, 255)]

                if classIds[i] == 0:  # Bike (person)
                    helmet_roi = img[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
                    c = helmet_or_nohelmet(helmet_roi)

                    if c == 1:
                        persons_without_helmet += 1
                    else:
                        persons_with_helmet += 1

                    persons_in_frame.append((x, y, x + w, y + h, c))  # Store person coordinates and helmet status

                if classIds[i] != 0:  # Vehicle (including license plate)
                    x_h = x - 60
                    y_h = y - 350
                    w_h = w + 100
                    h_h = h + 100
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)

                    if y_h > 0 and x_h > 0:
                        h_r = img[y_h:y_h + h_h, x_h:x_h + w_h]
                        c = helmet_or_nohelmet(h_r)
                        cv2.putText(img, ['helmet', 'no-helmet'][c], (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h), (255, 0, 0), 10)

                        if c == 1:
                            num_img = img[y:y + h, x:x + w]
                            numplate_filename = os.path.join(output_dir, f'{len(os.listdir(output_dir))}.jpg')
                            cv2.imwrite(numplate_filename, num_img)

                            # Save image name and detected character sequence to the list
                            unique_plate_data.append((os.path.basename(numplate_filename), ""))

        # Append the list of persons in the current frame to the persons list
        persons.append(persons_in_frame)

        # Draw boxes and labels on the persons in the current frame
        for person in persons_in_frame:
            x1, y1, x2, y2, c = person
            label = 'helmet' if c == 0 else 'no-helmet'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if c == 0 else (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if c == 0 else (0, 0, 255), 2)

        writer.write(img)
        cv2.imshow("License Plate Detection", img)

        if cv2.waitKey(1) == 27:
            break

    # Extract characters from saved number plate images and update the character sequences in unique_plate_data
    for i, filename in enumerate(os.listdir(output_dir)):
        if filename.endswith('.jpg'):
            numplate_image_path = os.path.join(output_dir, filename)
            numplate_image = cv2.imread(numplate_image_path)
            numplate_image_gray = cv2.cvtColor(numplate_image, cv2.COLOR_BGR2GRAY)

            # Apply OCR to extract characters
            extracted_text = pytesseract.image_to_string(numplate_image_gray, config='--psm 6')

            # Check if the character sequence has been seen before
            if extracted_text.strip() not in [seq[1] for seq in unique_plate_data]:
                for seq in unique_plate_data:
                    if seq[0] == os.path.basename(numplate_image_path):
                        seq = (seq[0], extracted_text.strip())
                        break

    # Save unique character sequences along with image names to a CSV file
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image Name", "Character Sequence"])
        csv_writer.writerows(unique_plate_data)

    # Release video resources and close windows
    writer.release()
    cap.release()
    cv2.destroyAllWindows()

# Create a main GUI window
root = tk.Tk()
root.title("License Plate Detection")

# Create and configure the GUI elements
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label = tk.Label(frame, text="Select Video File:")
label.pack()

video_file_entry = tk.Entry(frame)
video_file_entry.pack(fill="both", expand=True, padx=5, pady=5)

browse_button = tk.Button(frame, text="Browse", command=lambda: video_file_entry.insert(0, filedialog.askopenfilename()))
browse_button.pack()

start_button = tk.Button(frame, text="Start Processing", command=start_processing)
start_button.pack()

# Run the GUI application
root.mainloop()
