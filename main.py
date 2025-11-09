import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# Load a pretrained YOLO model
model = YOLO('yolo11s.pt')

# In-memory store for camera states to control the feed
camera_states = {}

async def generate_frames(camera_id: int):
    """
    Asynchronous generator function to yield video frames.
    It manages the camera resource within a try...finally block to ensure release.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open video device {camera_id}.")
        return

    camera_states[camera_id] = True
    try:
        while camera_states.get(camera_id):
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?).")
                break

            # Run YOLO inference
            results = model(frame)
            annotated_frame = results[0].plot()

            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
            if not flag:
                continue

            # Yield the output frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
            # Yield control to the event loop, making the server responsive
            await asyncio.sleep(0.01)
    finally:
        print(f"Releasing camera {camera_id}")
        cap.release()
        if camera_id in camera_states:
            del camera_states[camera_id]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page."""
    with open("index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    """Video streaming route."""
    return StreamingResponse(generate_frames(camera_id),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/stop_feed/{camera_id}")
async def stop_feed(camera_id: int):
    """Signals the generator to stop the video feed for a specific camera."""
    if camera_id in camera_states:
        camera_states[camera_id] = False
    return {"message": f"Stop signal sent to camera {camera_id}"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
