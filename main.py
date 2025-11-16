import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

# Create an instance of the FastAPI application
app = FastAPI(title="Car Brand Detector API")

# --- MODEL LOADING ---
# Load our custom model ONCE when the server starts up.
# This is a critical optimization to avoid slow load times on every request.
print("Loading Car Brand Detector model...")
model = torch.hub.load(
    'yolov5',  # Our local yolov5 folder
    'custom',
    path='yolov5/runs/train/car_brand_run4/weights/best.pt',  # Path to our trained weights
    source='local'
)
print("Model loaded successfully.")

# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    """Provides a simple welcome message."""
    return {"message": "Welcome! Send a POST request to /predict to detect car brands."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, runs detection, and returns the results.
    """
    # 1. Read the uploaded image file
    image_bytes = await file.read()
    
    # 2. Convert the bytes into an image that the model can understand
    # The 'RGB' conversion is important to ensure consistency
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 3. Run the model on the image
    results = model(img)

    # 4. Convert the results into a clean JSON format
    # .pandas().xyxy[0] gives a DataFrame of detections
    # .to_dict(orient="records") converts it to a list of dictionaries
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    # 5. Return the detections
    return {"detections": detections}