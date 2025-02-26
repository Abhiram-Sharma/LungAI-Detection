import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from PIL import Image
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load pretrained InceptionV3 model
model = models.inception_v3(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # Modify for 3 classes
model.eval()

# Define class labels
class_labels = ["Normal", "Benign", "Malignant"]

# Image preprocessing function
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 requires 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# SHAP explainer function
def generate_shap_explanation(image_tensor):
    explainer = shap.GradientExplainer(model, image_tensor)
    shap_values = explainer.shap_values(image_tensor)
    shap_img = np.abs(shap_values[0][0].transpose(1, 2, 0))
    shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min()) * 255
    return cv2.applyColorMap(shap_img.astype(np.uint8), cv2.COLORMAP_JET)

# API endpoint to predict and explain
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Generate SHAP heatmap
    shap_heatmap = generate_shap_explanation(image_tensor)

    # Save the heatmap
    shap_filename = f"shap_heatmap_{file.filename}.png"
    cv2.imwrite(shap_filename, shap_heatmap)

    return {
        "prediction": class_labels[predicted_class],
        "confidence": confidence,
        "shap_heatmap": shap_filename
    }

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
