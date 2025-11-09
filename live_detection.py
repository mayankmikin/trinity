import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model (e.g., yolo11s.pt)
# There is no official yolov11 in the ultralytics package.
# We will use yolo11s.pt for live object detection.
model = YOLO('yolo11s.pt')

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Set camera resolution (optional)
cap.set(3, 800)
cap.set(4, 600)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLOv11 Live Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
