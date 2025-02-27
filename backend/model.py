import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import numpy as np

# Define the number of classes (Normal, Benign, Malignant)
NUM_CLASSES = 3

def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Builds a deep learning model for lung cancer classification using ResNet50 as the base.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes (int): Number of output classes (default is 3).
    
    Returns:
        Model: Compiled Keras model ready for inference or training.
    """
    # Load pre-trained ResNet50 as the base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers (optional: unfreeze later for fine-tuning)
    base_model.trainable = False
    
    # Add custom layers on top of the base model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)  # Reduce spatial dimensions
    x = layers.Dense(128, activation='relu')(x)  # Fully connected layer
    x = layers.Dropout(0.5)(x)  # Regularization to prevent overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # Output layer
    
    # Create the full model
    model = Model(inputs, outputs, name='LungCancerClassifier')
    
    # Compile the model (use categorical_crossentropy for multi-class classification)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def load_trained_model(weights_path=None):
    """
    Loads a pre-trained model with custom weights if provided.
    
    Args:
        weights_path (str): Path to the .h5 file with trained weights (optional).
    
    Returns:
        Model: Loaded Keras model ready for inference.
    """
    model = build_model()
    if weights_path:
        try:
            model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}. Using default model.")
    return model

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses an input image for model inference.
    
    Args:
        image (PIL.Image): Input image to preprocess.
        target_size (tuple): Target size for resizing (height, width).
    
    Returns:
        np.ndarray: Preprocessed image array.
    """
    # Resize and convert to array
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(model, image):
    """
    Predicts the class of an input image using the provided model.
    
    Args:
        model (Model): Trained Keras model.
        image (PIL.Image): Input image to classify.
    
    Returns:
        dict: Prediction results with class label and probabilities.
    """
    # Preprocess the image
    img_array = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_labels = ['Normal', 'Benign', 'Malignant']
    class_prob = predictions[0][class_idx]
    
    return {
        'class': class_labels[class_idx],
        'probability': float(class_prob),
        'all_probabilities': predictions[0].tolist()
    }

if __name__ == "__main__":
    # Example usage (for testing purposes)
    model = load_trained_model()  # Optionally pass weights_path='path/to/lung_model.h5'
    model.summary()  # Print model architecture
    
    # Dummy test with a random image (replace with actual image loading for testing)
    from PIL import Image
    import numpy as np
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = predict_image(model, dummy_image)
    print("Prediction:", result)
