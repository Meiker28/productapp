
#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
from skimage import io
from skimage import img_as_ubyte
from skimage import transform
from keras.models import load_model
from keras.models import model_from_json

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
MODEL_ARCHITECTURE = './model/modelo.json'
MODEL_WEIGHTS = './model/peso.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    imagen = io.imread(img_path)
    imagen = img_as_ubyte(transform.resize(imagen, (imgsize, imgsize)))
    imagen = np.array(imagen, np.float32)
    imagen /= 255.
    imagen = imagen.reshape(1, 256, 256, 3)
    prediction = model.predict(imagen)
    prediction = np.argmax(prediction, axis= 1)


    return prediction

# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
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
        f.save(file_path)

        # Make a prediction
        prediction = model_predict(file_path, model)
        predicted_class = prediction

        #predicted_class = classes['TRAIN'][prediction[0]]
        #print('We think that is {}.'.format(predicted_class.lower()))

        return str(predicted_class).lower()


if __name__ == '__main__':
	app.run(debug = True)

