

from flask import Flask, render_template, request, render_template_string
import numpy as np  # Make sure to import necessary libraries for predictions
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



app = Flask(__name__)

# Placeholder for the predict_image function, replace with your actual implementation
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(256, 256), color_mode='grayscale')
    image_array = image.img_to_array(img)
    expand_image = np.expand_dims(image_array, axis=0)
    expand_image /= 255.0
    expand_image = expand_image.astype('float32')
    return expand_image

def predict_image(file_path):
    expand_image = preprocess_image(file_path)

    model = load_model("/Users/bibek/Desktop/app/resnetfinal_model1.h5")
    prediction = model.predict(expand_image)

    # Your image classification logic goes here
    # Replace this with your actual prediction code
    # Example: return np.array([0.2, 0.7, 0.1])
    print(prediction)
    print(file_path)

    return prediction

@app.route("/")
def home():
    return render_template('h.html', result=None)  # Pass result=None initially

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to a temporary location
    file_path = 'static/temp.jpg'
    file.save(file_path)

    # Make predictions
    prediction = predict_image(file_path)

    # Display the results
    classes = ['Benign', 'Malignant', 'Normal']
    result = classes[np.argmax(prediction)]

    # Use JavaScript to update the result dynamically
    return render_template('h.html', result=result, update_result=True)

if __name__ == "__main__":
    app.run(port=5001)
