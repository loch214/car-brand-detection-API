# Car Brand Detection API

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Framework](https://img.shields.io/badge/framework-FastAPI-green)
![Model](https://img.shields.io/badge/model-YOLOv5-orange)

This project is a complete, end-to-end machine learning application that identifies car brands from images. It uses a custom-trained YOLOv5 model, which has been fine-tuned to recognize 19 different car brands by learning their shapes and features, not just their logos. The final model is deployed as a high-performance web API using FastAPI.

This project was built step-by-step to bridge the gap between theoretical knowledge of the ML pipeline and a real-world, functional application that was iteratively improved based on performance analysis.

---

## Project Workflow

This project followed the complete end-to-end machine learning pipeline, including a full cycle of model analysis and improvement.

**1. Data Collection & Preparation:**
*   A high-quality dataset containing 19 car brands was sourced from Roboflow Universe.
*   The dataset was already pre-processed and labeled in the required YOLOv5 format.

**2. Model Training (Iterative Development):**
*   **V1 Model (Nov 16, 2025):** An initial model was trained for **50 epochs**. Analysis showed it was highly biased towards the 'Toyota' class due to an imbalanced dataset.
*   **V2 Model (Nov 18, 2025):** To address the bias, a second model was trained for **150 epochs** with **high-level data augmentation** (including random flips, rotations, and color changes). This forced the model to learn more robust features and significantly improved its overall accuracy.
*   All training was performed locally on an NVIDIA GeForce RTX 3050 6GB GPU, which was critical for accelerating the training process.

**3. Model Deployment:**
*   The final, improved V2 model weights (`best.pt`) were saved.
*   A web server was built using **FastAPI**, creating a `/predict` endpoint that accepts image uploads.
*   The API is served with Uvicorn, a high-performance ASGI server.

---

## Final Results & Analysis

The final V2 model, trained for 150 epochs with aggressive data augmentation, showed a significant improvement in overall accuracy (mAP) compared to the V1 model. It is now capable of identifying a wider range of brands.

#### **Prediction Example (Correct Prediction):**
The V2 model correctly identifies a **Toyota** Prius.

![Correct Toyota Prediction](assets/toyota_prediction.png) (will add the image later)

#### **Analysis of Model Limitations:**
Despite the improvements, further testing revealed the model's limitations. When given an image of a white BMW SUV, the model incorrectly predicted it as `kia` and `hyundai`, albeit with very low confidence scores (19% and 18%).

![Incorrect BMW Prediction](assets/bmw_prediction.png)

**Diagnosis:** This result stems from two core challenges:

1.  **Limited Dataset Size:** The training dataset contains only **~1700 images**. For a complex task with 19 visually similar classes, this is a relatively small dataset. The model has not seen enough examples of each brand in various conditions to build a highly confident understanding.

2.  **Low Inter-class Variance:** The model's confusion is most apparent between brands with similar design languages (e.g., modern SUVs). The small dataset size makes it very difficult for the model to learn the subtle differences between these classes. The low confidence scores are a direct indicator of this uncertainty.

**Future Improvements:**
Based on this analysis, the clear path to a production-ready V3 model involves a data-centric approach:

*   **Significant Data Sourcing:** The most critical step is to expand the dataset. A much larger and more diverse set of images (ideally 1,000+ per brand) would be required for the model to learn the subtle, distinguishing features and improve both accuracy and confidence.
*   **Further Training:** With a larger dataset, training for more epochs (200+) would become more effective and yield better results.

This project demonstrates a real-world understanding of the iterative nature of ML development: build, test, analyze the root cause of limitations, and define a clear path for improvement.
---

## Technology Stack

*   **Backend:** FastAPI, Uvicorn
*   **AI / Machine Learning:** PyTorch, YOLOv5
*   **Core Python Libraries:** Pillow, python-multipart

---

## How to Run This Project

Follow these steps to set up and run the project locally.

**1. Clone the repository:**

2. Create and Activate a Virtual Environment:
This project requires Python 3.11.
code
Bash
py -3.11 -m venv venv-gpu
.\venv-gpu\Scripts\activate
3. Install Dependencies:
code
Bash
pip install torch torchvision torau`dio --index-url https://download.pytorch.org/whl/cu121
pip install -r yolov5/requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart Pillow
4. Train Your Own Model (Optional):
Download the dataset from Roboflow and place it in the root folder. Then, update data.yaml with the correct local paths and run the training command from within the yolov5 folder.
5. Run the API Server:
(You must first have a trained best.pt file from training and update the path in main.py)
code
Bash
uvicorn main:app --reload```

**6. Test the API:**
Open your browser to `http://127.0.0.1:8000/docs` and use the interactive interface to upload an image to the `/predict` endpoint.
```bash
git clone https://github.com/your-username/car-brand-detection-API.git
cd car-brand-detection-API
