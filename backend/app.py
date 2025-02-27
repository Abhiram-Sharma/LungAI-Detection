from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import shap
import base64
from io import BytesIO

app = Flask(__name__)

# Load a pre-trained model (replace with your model)
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
model = tf.keras.Sequential([model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(3, activation='softmax')])
# Assume model is trained and saved as 'lung_model.h5'
# model.load_weights('lung_model.h5')

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_heatmap(image, model):
    explainer = shap.GradientExplainer(model, np.zeros((1, 224, 224, 3)))
    shap_values = explainer.shap_values(image)
    heatmap = np.mean(shap_values[0], axis=-1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img_processed = preprocess_image(img)

    # Predict
    prediction = model.predict(img_processed)
    classes = ['Normal', 'Benign', 'Malignant']
    result = classes[np.argmax(prediction)]

    # Generate heatmap
    heatmap = generate_heatmap(img_processed, model)
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
    buffered = BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'classification': result, 'heatmap': heatmap_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
