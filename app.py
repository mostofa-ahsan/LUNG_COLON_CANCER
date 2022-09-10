from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import tensorflow
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "model_60_type_2"

# Load your trained model
# model = load_model(MODEL_PATH)
import tensorflow_text as text
imported_60_type_2 = tensorflow.keras.models.load_model(MODEL_PATH)


# import numpy as np
# from keras.preprocessing.image import load_img, img_to_array
# def preprocess_image(path):
#     img = load_img(path, target_size = (img_height, img_width))
#     a = img_to_array(img)  # δημιούργησε τον πίνακα με τις τιμές της εικόνας
#     a = np.expand_dims(a, axis = 0)  # μετέτρεψε τον τρισδιάστατο πινακα σε τετραδιάστατο
#     a /= 255.  # επανακλιμάκωσε τις τιμές στο διάστημα [0, 1]
#     return a

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = imported_60_type_2.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "COLON ACA"
    elif preds == 1:
        preds = "COLON BENIGN"
    elif preds == 2:
        preds = "LUNG ACA"
    elif preds == 3:
        preds = "LUNG BENIGN"
    else:
        preds = "LUNG SCC"

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path.replace("\\","/"))

        # Make prediction
        preds = model_predict(file_path, imported_60_type_2)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)