import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Activation, 
    Convolution2D, 
    Dropout, 
    Conv2D, 
    MaxPool2D,
    AveragePooling2D, 
    BatchNormalization, 
    Flatten, 
    GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
# Flask utils
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Define a flask app
COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('uploads/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('uploads/{}.jpg'.format(COUNT))
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    ary = Image.fromarray(img_arr, 'RGB')
    r = np.array(ary.resize((50,50)))
    print(r.shape)
    
    model = load_model("weights.h5")
    la = LabelEncoder()
    train_labels = np.load("encoded_labels.npy")
    labels = pd.DataFrame(train_labels)
    train_labels_encoded = la.fit_transform(labels[0])
    
    r = np.expand_dims(r, axis=0)
    pred_t = np.argmax(model.predict(r), axis=1)
    prediction_t = la.inverse_transform(pred_t)

    COUNT += 1
    return render_template('prediction.html', data=prediction_t[0])


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('uploads', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)
