import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load an image
image = cv2.imread("inputs\car.jpg")
height, width = image.shape[:2]

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set input to the network
net.setInput(blob)

# Get the output layer names
layer_names = net.getUnconnectedOutLayersNames()

# Run forward pass to get object detection predictions
outputs = net.forward(layer_names)

# Initialize lists to store bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Define confidence threshold
conf_threshold = 0.5

# Define non-maximum suppression threshold
nms_threshold = 0.4

# Loop through each output
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > conf_threshold:
            # Scale the bounding box coordinates back to the original image
            box = detection[0:4] * np.array([width, height, width, height])
            (centerX, centerY, box_width, box_height) = box.astype("int")
            
            # Calculate the top-left corner of the bounding box
            x = int(centerX - (box_width / 2))
            y = int(centerY - (box_height / 2))
            
            # Add bounding box coordinates, confidences, and class IDs to lists
            boxes.append([x, y, int(box_width), int(box_height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes on the image
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        
        # Draw bounding box and label on the image
        color = (0, 255, 0)  # BGR color format (green)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the resulting image to a file
cv2.imwrite("detection_result.jpg", image)
