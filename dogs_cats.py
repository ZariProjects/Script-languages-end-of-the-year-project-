import base64
import numpy as np 
import io
import os
from PIL import Image
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from flask import request
from flask import send_from_directory
from flask import redirect
from flask import jsonify
from flask import Flask
from flask import url_for
from flask import render_template

from werkzeug.utils import secure_filename

app = Flask(__name__)


vgg16_model = keras.applications.VGG16(
    input_shape=[224, 224, 3],
    weights='imagenet',
    include_top=False
)
for layer in vgg16_model.layers:
    layer.trainable = False  
output_layer = Dense(2, activation='softmax')(Flatten()(vgg16_model.output))
model = Model(inputs=vgg16_model.input, outputs=output_layer)

def get_model():
    global model
    model = load_model("dogs_cats_prediction_model.h5")
    model.compile(
        Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model Loaded Successfully")
    

def preprocess_image(image, target_size):
    image = Image.open(image)
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPG"]

def allowed_images(filename):
    if filename == "":
        return False
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/upload/<filename>")
def transfer_images(filename):
    return send_from_directory("images", filename)

@app.route("/upload_and_predict_image", methods=['POST'])
def upload():
    error_statement = ""
    if request.method == "POST":
        if request.files['image'].filename:
            image = request.files['image']
            if allowed_images(image.filename) == False:
                error_statement = "Image has invalid size, name or extension." 
                return render_template("index.html", error_statement=error_statement, prediction="", dog_prediction = "", cat_prediction = "")
            else:
                filename = secure_filename(image.filename)
                error_statement = ""
                procesed_image = preprocess_image(image, target_size=(224, 224))
                get_model()
                prediction = model.predict(procesed_image).tolist()
                dog_prediction = float(prediction[0][0])
                cat_prediction = float(prediction[0][1])
                return render_template("index.html", prediction=prediction, dog_prediction=dog_prediction, cat_prediction=cat_prediction)
        else:
            error_statement="No image has been selected."
            return render_template("index.html", error_statement=error_statement, prediction="", dog_prediction = "", cat_prediction = "")
            
@app.route('/')
def running():
    return render_template('index.html', prediction="", dog_prediction = "", cat_prediction = "")

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=False)
    
