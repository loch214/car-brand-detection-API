print("--- SCRIPT STARTED ---") # <-- ADD THIS LINE

import torch

# Step 1: Load the pre-trained YOLOv5 model from the internet
# 'yolov5s' is the small, fast version. 'pretrained=True' is the key part.
print("Downloading/Loading YOLOv5 model... (This might take a moment on first run)")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded successfully.")

# Step 2: Define the path to your image
image_path = 'test_image.jpg' 

# Step 3: Run the model on your image to detect objects
print(f"Running object detection on {image_path}...")
results = model(image_path)
print("Detection complete.")

# Step 4: Print the detected objects in a clean table format
# The 'results' object contains all the information. We can access it as a pandas DataFrame.
print("\n----------- DETECTED OBJECTS -----------")
detected_objects_df = results.pandas().xyxy[0]
print(detected_objects_df)
print("----------------------------------------")

# Step 5 (Optional but cool): Show the image with bounding boxes drawn on it
# This will open a new window displaying your image. Close the window to end the script.
print("\nDisplaying image with detected objects. Close the image window to exit.")
results.show()