import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('models/animal_cnn_model.keras')

# Class labels (these should correspond to your training directories)
class_labels = ['bear_png', 'chinkara', 'elephant', 'lion', 'peacock', 'pig', 'sheep', 'tiger']

def predict_animal(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    
    # Print confidence scores for all classes
    print(f"Confidence scores for each class: {dict(zip(class_labels, predictions[0]))}")
    
    return class_labels[predicted_class], predictions

if __name__ == '__main__':
    image_path = 'data/train/peacock/001.jpg'  # Provide the path to the test image
    animal_class, confidence = predict_animal(image_path)
    print(f"Predicted Animal Class: {animal_class} with confidence: {confidence}")
