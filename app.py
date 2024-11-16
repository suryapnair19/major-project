from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import pickle

app = Flask(__name__)

# Load the trained model
with open('modelface.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is in the "templates" folder

# Define a function to preprocess images
def preprocess_image(image):
    # Convert image to grayscale (single channel)
    image = image.convert('L')  # 'L' mode is for grayscale
    image = image.resize((50, 50))  # Assuming model expects 50x50 images; adjust if needed
    image = np.array(image) / 255.0  # Normalize to [0, 1] if required by model
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (shape: (50, 50, 1))
    image = np.expand_dims(image, axis=0)  # Add batch dimension (shape: (1, 50, 50, 1))
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Predict using the model
        prediction = model.predict(processed_image)

        # Ensure the prediction is in the expected format
        class_label = 'Masked' if prediction[0][0] > 0.5 else 'Unmasked'

        # Return the prediction as JSON
        return jsonify({"prediction": class_label})
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    app.run(debug=True)