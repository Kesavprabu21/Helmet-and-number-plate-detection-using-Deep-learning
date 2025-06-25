import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt
import glob
import os
import pytesseract
import csv

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Set GPU allow growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load YOLO model
net = cv2.dnn.readNet("D:/projects/Helmet-detection-and-number-plate-Extraction-main/yolov3-custom_7000.weights", "D:/projects/Helmet-detection-and-number-plate-Extraction-main/yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load helmet detection model
model = load_model("D:/projects/Helmet-detection-and-number-plate-Extraction-main/helmet-nonhelmet_cnn.h5")
print('model loaded!!!')

# Open video capture
cap = cv2.VideoCapture("D:/projects/Helmet-detection-and-number-plate-Extraction-main/video.mp4")

# Define colors
COLORS = [(0, 255, 0), (0, 0, 255)]

# YOLO layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5, (888, 500))

# Function to classify helmet or no-helmet
def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        return int(model.predict(helmet_roi)[0][0])
    except:
        pass

ret = True
plates = []
l = 0
hel_count = 0
non_hel_count = 0
path = "D:/projects/Helmet-detection-and-number-plate-Extraction-main/numberplates"

while ret:
    ret, img = cap.read()
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

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]
            if classIds[i] == 0:  # bike
                helmet_roi = img[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
            else:  # number plate
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
                        cv2.imwrite(os.path.join(path, str(l) + '.jpg'), num_img)
                        l = l + 1
                        non_hel_count = non_hel_count + 1
                    if c == 0:
                       hel_count = hel_count + 1
                       num_img = img[y:y + h, x:x + w]
                    p_r = pytesseract.image_to_string(num_img, lang='eng', config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    filter_p_r = "".join(p_r.split()).replace("\n", " ")
                    plates.append(filter_p_r)

    writer.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

writer.release()
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Total No. of riders: ", hel_count + non_hel_count)
print("No. of riders with Helmet: ", hel_count)
print("No. of riders without Helmet: ", non_hel_count)

# Function to preprocess an image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Perform preprocessing operations (e.g., resizing, cropping, enhancing)

    return image

# Function to extract license plate text from an image
def extract_license_plate_text(image):
    # Use pytesseract to perform OCR on the image
    text = pytesseract.image_to_string(image, config='--psm 7')  # Adjust configuration as needed

    return text.strip()  # Remove leading/trailing whitespace

# Function to process images in a folder
def process_images_in_folder(folder_path):
    data = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Add more file extensions as needed
            image_path = os.path.join(folder_path, filename)
            preprocessed_image = preprocess_image(image_path)
            license_plate = extract_license_plate_text(preprocessed_image)

            data.append({'image_path': image_path, 'license_plate': license_plate})

    return data

# Function to save data to a CSV file
def save_to_csv(data, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Image_Path', 'License_Plate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item in data:
            writer.writerow({'Image_Path': item['image_path'], 'License_Plate': item['license_plate']})

if __name__ == '__main__':
    folder_path = 'D:/projects/Helmet-detection-and-number-plate-Extraction-main/numberplates'  # Replace with your folder path
    csv_filename = 'license_plate_data.csv'

    data = process_images_in_folder(folder_path)
    save_to_csv(data, csv_filename)

