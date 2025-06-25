# Helmet-detection-and-number-plate-Extraction
This abstract presents a powerful video analysis solution that seamlessly combines cutting-edge technologies for enhanced safety and actionable insights. By integrating YOLO object detection and a custom CNN model, the system rapidly identifies bikes, assesses helmet usage, and even extracts alphanumeric details from number plates. The YOLO algorithm efficiently detects objects of interest, while the CNN model precisely classifies helmet presence. Leveraging OCR, the system extracts vehicle identification from number plates. Real-time visual overlays of bounding boxes, helmet labels, and extracted values offer immediate situational awareness. This technology holds great potential for safety enforcement, traffic monitoring, and data-driven decision-making. The demonstrated framework showcases the fusion of object detection, classification, and data extraction, creating a versatile solution with applications across diverse industries.


Technical Flowchart
![image](https://github.com/user-attachments/assets/59c54b55-2197-4faf-8547-03168e77d22a)

 

Description
Initialization and Setup:
The code begins by initializing essential libraries and environment settings. It sets up the TensorFlow environment with GPU growth enabled and loads the YOLO (You Only Look Once) object detection model, configured to utilize CUDA acceleration for optimal performance. Additionally, the custom helmet classification Convolutional Neural Network (CNN) model is loaded, bringing an AI-powered helmet detection capability.
Video Source and Loop:
The system opens a video source (specified as 'video.mp4') to process frames sequentially. Inside a loop, each frame is read, and preprocessing for YOLO is performed. This involves converting the frame into a format suitable for YOLO's input.

Object Detection:
The YOLO model is utilized for object detection within each frame. It efficiently identifies objects of interest, particularly bikes and number plates, based on predefined criteria. Detected objects are enclosed within bounding boxes, and their class probabilities are analyzed.

Helmet Classification:
For each detected bike, a region of interest (ROI) around the rider's head is cropped. This ROI undergoes helmet classification using the pre-trained custom CNN model. The model analyzes whether a helmet is worn by the rider and assigns a classification label (helmet or no-helmet).

Number Plate Extraction and OCR:
When a number plate is detected, a distinct region is isolated. Optical Character Recognition (OCR) using the Tesseract engine is applied to extract alphanumeric characters from the number plate region. This allows for automatic reading of vehicle identification details.

Visualization and Display:
Visual feedback is essential, and the processed frames are displayed in real-time. Bounding boxes around detected objects, helmet classification labels, and extracted number plate values are superimposed onto the frame. These visual indicators enhance situational awareness and facilitate decision-making.

Loop Continuation and Key Press:
The processing loop continues until all frames are analyzed. A check for a key press is implemented, enabling the user to terminate the process (usually using the 'ESC' key).

Video Output:
The processed frames, including overlays and visual indicators, are compiled into an output video ('output.avi') using the VideoWriter module. This video serves as a comprehensive representation of the object detection, helmet classification, and number plate extraction processes.

Conclusion:
In summary, the code demonstrates a sophisticated video analysis system that integrates object detection, helmet classification, and number plate extraction using YOLO, a custom CNN model, and OCR techniques. The resulting visual feedback enhances safety monitoring, classification accuracy, and data extraction from video streams, making the solution applicable across various domains requiring enhanced situational awareness and decision-making.








