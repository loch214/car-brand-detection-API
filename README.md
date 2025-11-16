# Car Brand Detection API

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Framework](https://img.shields.io/badge/framework-FastAPI-green)
![Model](https://img.shields.io/badge/model-YOLOv5-orange)

This project is a complete, end-to-end machine learning application that identifies car brands from images. It uses a custom-trained YOLOv5 model, which has been fine-tuned to recognize 19 different car brands by learning their shapes and features, not just their logos. The final model is deployed as a high-performance web API using FastAPI.

This project was built step-by-step to bridge the gap between theoretical knowledge of the ML pipeline and a real-world, functional application.

## Live Demo & Results

The final API successfully loads the custom model and can make predictions on new images it has never seen before.

#### **Prediction Example:**
The model correctly identifies a **Toyota** Prius from a test image.

![Detection Result](assets/detection_result.png)

As shown, the model correctly identifies the brand as "toyota". The confidence score is around 0.27, which is a solid initial result. This lower confidence is expected because the model was trained for a limited **50 epochs** on a dataset of around 1,700 training images. With more training time or a larger dataset, this confidence score would improve significantly.

---

## Project Workflow

This project followed the complete end-to-end machine learning pipeline:

**1. Data Collection & Preparation:**
*   A high-quality dataset containing 19 car brands was sourced from Roboflow Universe.
*   The dataset was already pre-processed and labeled in the required YOLOv5 format, which includes images and corresponding text files with bounding box coordinates.

**2. Model Training (Fine-Tuning):**
*   We started with the official, pre-trained `yolov5s` model, which knows how to detect 80 common objects.
*   This model was then **fine-tuned** on the custom car brand dataset for 50 epochs.
*   The training was performed locally on an **NVIDIA GeForce RTX 3050 6GB GPU**, which accelerated the process from several hours (on CPU) to just **30 minutes**.

**3. Model Deployment:**
*   The final trained model weights (`best.pt`) were saved.
*   A web server was built using **FastAPI**.
*   The API loads the custom model on startup and provides a `/predict` endpoint that accepts image uploads.
*   The server is run using Uvicorn, a high-performance ASGI server.

---

## Technology Stack

*   **Backend:** FastAPI, Uvicorn
*   **AI / Machine Learning:** PyTorch, YOLOv5
*   **Core Python Libraries:** Pillow (for image manipulation), python-multipart (for file uploads)

---

## How to Run This Project

Follow these steps to set up and run the project locally.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/car-brand-detection-API.git
cd car-brand-detection-API
