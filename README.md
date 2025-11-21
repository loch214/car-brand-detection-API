# Car Brand Detection API

This is a personal project I built to learn the end-to-end process of creating a machine learning application. My goal was to train a computer vision model to identify car brands and then deploy it as a web API.

This was a learning experience, and the final model is **not a production-ready, high-accuracy tool.** The true goal was to understand the complete pipeline, from data to a live server.

---

## What I Learned

*   **The Full ML Pipeline:** I successfully completed all the steps: sourcing a dataset, fine-tuning a YOLOv5 model, and deploying it with FastAPI.
*   **CPU vs. GPU:** I experienced the massive speed difference between training on a CPU and a GPU (NVIDIA RTX 3050), which took the training time from hours to minutes.
*   **Model Limitations:** My biggest takeaway was how much model performance depends on the dataset. My final model struggles because the dataset was small (~1700 images for 19 classes) and imbalanced. I learned to diagnose this by testing the model and analyzing its weak points.
*   **API Development:** I learned how to build a simple, functional REST API using Python and FastAPI to serve my trained model.

---

## The Process

The entire training journey, including the code I used and my analysis, is documented in the `training_log.ipynb` notebook.

1.  **V1 Training:** I first trained a model for 50 epochs and discovered it was heavily biased.
2.  **V2 Training:** I retrained a second model for 150 epochs with aggressive data augmentation to create a better, more robust model.
3.  **Deployment:** The final V2 model was then deployed in a FastAPI application.

---

## How to Run It

This project uses Python 3.11. The model was trained using the `yolov5` submodule.

**1. Setup Environment:**
Create a virtual environment and install dependencies. The main libraries are PyTorch, YOLOv5's requirements, and FastAPI.

**2. Run the Server:**
The `main.py` file contains the FastAPI application. To run it, use uvicorn:
```bash
uvicorn main:app --reload