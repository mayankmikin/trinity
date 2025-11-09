from ultralytics import YOLO

# Load a pretrained YOLO model (e.g., yolov8n.pt, yolov9c.pt)
# There is no official yolov11 in the ultralytics package.
# You can choose from models like: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
# or the newer yolov9c.pt
model = YOLO('yolo11s.pt')

# Define the image to run the prediction on
# You can use a local file path or a URL
image_path = 'https://ultralytics.com/images/bus.jpg'

# Run the prediction
results = model(image_path)

# Print the results
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to file
