import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI(title="Car Brand Detector API")

# --- MODEL LOADING ---
print("Loading Car Brand Detector model...")
model = torch.hub.load(
    'yolov5',
    'custom',
    path='yolov5/runs/train/car_brand_v2_150_epochs2/weights/best.pt', # Make sure this is your V2 model
    source='local'
)
print("Model loaded successfully.")


# --- THIS IS THE FIX ---
# Set the confidence threshold on the model itself, right after loading.
# This is the correct way to configure the model's behavior.
model.conf = 0.1  # Set the confidence threshold to 10%
print(f"Model confidence threshold set to {model.conf}")


# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Welcome! Send a POST request to /predict to detect car brands."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Now, we just run the prediction without the extra argument
    results = model(img)

    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return {"detections": detections}