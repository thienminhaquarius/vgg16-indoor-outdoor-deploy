# end web sevices
from flask import request
from flask import jsonify
from flask import Flask, render_template
from flask_cors import CORS

# Image Processing
import os
import base64
import numpy as np
import io
import PIL.Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import pickle
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
import gc

# memory
# import psutil

app = Flask(__name__)
CORS(app)


def loadDataFromDisk(modelSavePath):
    data = pickle.load(open(modelSavePath, 'rb'))
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encodedImg = message['imgStr']
    decodedImg = base64.b64decode(encodedImg)

    # We got our image here
    img = image.load_img(io.BytesIO(decodedImg), target_size=(224, 224))

    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    model = VGG16(
        weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

    # Get feature
    feature = model.predict(img_data)
    feature = np.ndarray.flatten(feature)
    feature = np.ndarray.flatten(feature)
    feature = np.expand_dims(feature, axis=0)

    # clear data cho lần predict trước đó
    K.clear_session()

    # SVM 800 Image dataset
    modelSvm = loadDataFromDisk('./model.dat')

    # Predict label
    label = modelSvm.predict(feature)
    result = ''
    if label[0] == 0:
        result = 'Indoor'
    else:
        result = 'Outdoor'

    response = {
        "prediction": result,
    }

    # show memory
    # process = psutil.Process(os.getpid())
    # print('Ram: ', process.memory_info().rss/1000000)  # in bytes

    # clear memory
    del message
    del encodedImg
    del decodedImg
    del img
    del img_data
    del model
    del feature
    del modelSvm
    del result
    gc.collect()

    return jsonify(response)


if __name__ == "__main__":
    app.run()
