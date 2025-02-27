from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import shap
import base64
from io import BytesIO
from model import load_trained_model, preprocess_image, predict_image

app = Flask(__name__)

# Load the model (replace with your weights path if available)
model = load_trained_model()  # Optionally: load_trained_model('path/to/lung_model.h5')

def generate_heatmap(image, model):
    """
    Generates a SHAP heatmap for the input image.
    """
    img_array = preprocess_image(image)
    explainer = shap.GradientExplainer(model, np.zeros((1, 224, 224, 3)))
    shap_values = explainer.shap_values(img_array)
    heatmap = np.mean(shap_values[0], axis=-1)  # Aggregate across channels
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
    return heatmap

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    
    # Predict class
    prediction = predict_image(model, img)
    
    # Generate heatmap
    heatmap = generate_heatmap(img, model)
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
    buffered = BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Return results
    return jsonify({
        'classification': prediction['class'],
        'probability': prediction['probability'],
        'heatmap': heatmap_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
