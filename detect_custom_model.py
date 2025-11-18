import torch

# Load our local model from our local yolov5 code folder
print("Loading your custom Car Brand Detector model...")
model = torch.hub.load(
    'yolov5',  # The path to our local yolov5 folder
    'custom',  # We're loading a custom model
    path='yolov5/runs/train/car_brand_run4/weights/best.pt',  # Path to our trained weights
    source='local'  # Load from local files, do not use internet
)
print("Custom model loaded successfully.")

# Define the path to your new test image
image_path = 'car_test1.jpg'  # Make sure this is in your main project folder

# Run detection
print(f"Detecting brands in {image_path}...")
results = model(image_path)
print("Detection complete.")

# Print results in a table
print("\n----------- DETECTED CAR BRANDS -----------")
detected_objects_df = results.pandas().xyxy[0]
print(detected_objects_df)
print("-----------------------------------------")

# Show the image with the bounding box and brand name!
print("\nDisplaying image with detected brands. Close the image window to exit.")
results.show()