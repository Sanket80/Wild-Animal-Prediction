import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('models/animal_cnn_model.keras')

# Class labels
class_labels = ['bear_png', 'chinkara', 'elephant', 'lion', 'peacock', 'pig', 'sheep', 'tiger']

# Function to process and predict image class
def predict_animal(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    
    return class_labels[predicted_class], predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No file selected", 400
    
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        # Predict the class
        predicted_class, confidence = predict_animal(filepath)
        
        response = {
            'predicted_class': predicted_class,
            'confidence_scores': {class_labels[i]: float(confidence[0][i]) for i in range(len(class_labels))}
        }
        
        return jsonify(response)

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
