# -*- coding: utf-8 -*-
"""
Updated on Thu Mar 19 2024 for Fingerprint Recognition

@author: Your Name
"""

from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
# Path to save the uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model data from the pickle files
MODEL_PATH_ID = 'model_pickle.pkl'
MODEL_PATH_FINGER = 'model_pickle1.pkl'

with open(MODEL_PATH_ID, 'rb') as file:
    loaded_model_data_id = pickle.load(file)

with open(MODEL_PATH_FINGER, 'rb') as file:
    loaded_model_data_finger = pickle.load(file)

# Reconstruct the models from the saved data
model_id = tf.keras.models.model_from_json(loaded_model_data_id['config'])
model_id.set_weights(loaded_model_data_id['weights'])

model_finger = tf.keras.models.model_from_json(loaded_model_data_finger['config'])
model_finger.set_weights(loaded_model_data_finger['weights'])

# Function to make predictions
def model_predict(img_path, model_id, model_finger):
    img = image.load_img(img_path, target_size=(96, 96), color_mode='grayscale')
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    y_SubjectID_pred = model_id.predict(x)
    Id_pred = np.argmax(y_SubjectID_pred)

    y_fingerNum_pred = model_finger.predict(x)
    fingerNum_pred = np.argmax(y_fingerNum_pred)
    return Id_pred, fingerNum_pred

# Function to determine the finger name
def show_fingername(fingernum):
    if fingernum >= 5:
        fingername = "right "
        fingernum -= 5
    else:
        fingername = "left "
    if fingernum == 0:
        fingername += "little"
    elif fingernum == 1:
        fingername += "ring"
    elif fingernum == 2:
        fingername += "middle"
    elif fingernum == 3:
        fingername += "index"
    else:
        fingername += "thumb"
    return fingername

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to the uploads folder
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        # Make prediction
        Id_pred, fingerNum_pred = model_predict(file_path, model_id, model_finger)

        # Example condition to check prediction (update with your logic)
        if Id_pred == 527 and fingerNum_pred == 7:
            result_text = f"Information confirm! Fingerprint matches: person ID {Id_pred}, Finger: {show_fingername(fingerNum_pred)}. You belong to this organization."
        else:
            result_text = "Oops! You need authentication to enter here.person ID {Id_pred}, Finger: {show_fingername(fingerNum_pred)}"

        # Send image file path, and result_text to the template
        return render_template('index.html', user_image=filename, result=result_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
