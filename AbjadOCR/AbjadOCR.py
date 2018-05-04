import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import json
import glob
import os
import time
import uuid
from scipy.misc import imsave, imread, imresize
from keras.utils import np_utils
from arabocr.datasets import AbjadData
from arabocr.models import AbjadKeras
from arabocr.preprocessing import AbjadImage
from arabocr.utils import AbjadUtils
from html.parser import HTMLParser
from flask import Flask, render_template, request, url_for, send_file

# fields
img_width, img_height = 32, 32
model = None
labels = []
fit_and_save = False

# initialize the app from Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/word_recog')
def word_recog():
    return render_template('word_recog.html')

@app.route('/' + 'predict' + '/', methods = ['GET', 'POST'])
def predict():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = np.invert(out)

    plt.imshow(out, cmap='gray')
    plt.show()

    #USE CONTOUR TRICK HERE...

    out = imresize(out, (img_width, img_height))
    out = out.reshape(1, img_width, img_height, 1)
    prediction = AbjadKeras.get_prediction(model, out, labels)

    return prediction

@app.route('/' + 'confidences' + '/', methods = ['GET', 'POST'])
def confidences():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = np.invert(out)
    out = imresize(out, (img_width, img_height))
    out = out.reshape(1, img_width, img_height, 1)

    #USE CONTOUR TRICK HERE...

    dict_confz = AbjadKeras.get_prediction_confidences(model, out, labels)
    json_dict = json.dumps({'confz': dict_confz}, cls = AbjadUtils.CollectionEncoder)

    with tf.get_default_graph().as_default():
        response = json_dict
        return response

@app.route('/' + 'deskew_image' + '/', methods = ['GET', 'POST'])
def deskew_image():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = AbjadImage.invert_binary(out)

    out = AbjadImage.deskew_image(out)

    out = AbjadImage.invert_binary(out)
    out = AbjadImage.clamp01(out)
    out_file_path = 'static/outputs/deskew' + str(uuid.uuid4()) + '.png'
    imsave(out_file_path, out)
    return out_file_path

@app.route('/' + 'remove_dots_image' + '/', methods = ['GET', 'POST'])
def remove_dots_image():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = AbjadImage.invert_binary(out)

    out = AbjadImage.deskew_image(out)
    out = AbjadImage.binarize(out)
    out, _ = AbjadImage.remove_dots_diacritics(out)

    out = AbjadImage.invert_binary(out)
    out = AbjadImage.clamp01(out)

    out_file_path = 'static/outputs/remove_dots' + str(uuid.uuid4()) + '.png'
    imsave(out_file_path, out)
    return out_file_path
    
@app.route('/' + 'seperate_cc' + '/', methods = ['GET', 'POST'])
def seperate_connected_components():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = AbjadImage.invert_binary(out)

    out = AbjadImage.deskew_image(out)
    out = AbjadImage.binarize(out)
    out, dots_diacritics_mask = AbjadImage.remove_dots_diacritics(out)
    out = AbjadImage.seperate_connected_components(out, dots_diacritics_mask)

    out = AbjadImage.invert_binary(out)
    out = AbjadImage.clamp01(out)

    out_file_path = 'static/outputs/seperate_subwords' + str(uuid.uuid4()) + '.png'
    imsave(out_file_path, out)
    return out_file_path

@app.route('/' + 'word_baseline' + '/', methods = ['GET', 'POST'])
def word_baseline():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = AbjadImage.invert_binary(out)

    out = AbjadImage.deskew_image(out)
    out = AbjadImage.binarize(out)
    out, dots_diacritics_mask = AbjadImage.remove_dots_diacritics(out)
    out = AbjadImage.seperate_connected_components(out, dots_diacritics_mask)
    out, rgb, _, _ = AbjadImage.word_baseline(out)

    rgb = AbjadImage.invert_binary(rgb)
    out = AbjadImage.clamp01(out)

    out_file_path = 'static/outputs/word_baseline' + str(uuid.uuid4()) + '.png'
    imsave(out_file_path, rgb)
    return out_file_path

@app.route('/' + 'v_line_subwords_seperator' + '/', methods = ['GET', 'POST'])
def v_line_subwords_seperator():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = AbjadImage.invert_binary(out)

    out = AbjadImage.deskew_image(out)
    out = AbjadImage.binarize(out)
    out, dots_diacritics_mask = AbjadImage.remove_dots_diacritics(out)
    out = AbjadImage.seperate_connected_components(out, dots_diacritics_mask)
    out, rgb = AbjadImage.v_line_subwords_seperator(out)

    rgb = AbjadImage.invert_binary(rgb)
    out = AbjadImage.clamp01(out)

    out_file_path = 'static/outputs/v_line_subwords_seperator' + str(uuid.uuid4()) + '.png'
    imsave(out_file_path, rgb)
    return out_file_path

@app.route('/' + 'get_skeleton' + '/', methods = ['GET', 'POST'])
def get_skeleton():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = AbjadImage.invert_binary(out)

    out = AbjadImage.deskew_image(out)
    out = AbjadImage.binarize(out)
    out, dots_diacritics_mask = AbjadImage.remove_dots_diacritics(out)
    out = AbjadImage.seperate_connected_components(out, dots_diacritics_mask)
    out = AbjadImage.get_skeleton(out)

    out = AbjadImage.invert_binary(out)
    out = AbjadImage.clamp01(out)

    out_file_path = 'static/outputs/get_skeleton' + str(uuid.uuid4()) + '.png'
    imsave(out_file_path, out)
    return out_file_path

@app.route('/' + 'psp_seg' + '/', methods = ['GET', 'POST'])
def psp_segmentation():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = AbjadImage.invert_binary(out)

    #out = AbjadImage.deskew_image(out)
    out = AbjadImage.binarize(out)
    out, dots_diacritics_mask = AbjadImage.remove_dots_diacritics(out)
    out = AbjadImage.seperate_connected_components(out, dots_diacritics_mask)
    out = AbjadImage.get_skeleton(out)
    out, rgb, _ = AbjadImage.psp_segmentation(out)

    rgb = AbjadImage.invert_binary(rgb)
    out = AbjadImage.invert_binary(out)
    out = AbjadImage.clamp01(out)

    out_file_path = 'static/outputs/psp_segmentation' + str(uuid.uuid4()) + '.png'
    imsave(out_file_path, rgb)
    return out_file_path

@app.route('/' + 'predict_word' + '/', methods = ['GET', 'POST'])
def predict_word():
    img_data = request.get_data()
    out = AbjadImage.convert_from_Flask(img_data)
    out = AbjadImage.invert_binary(out)

    #out = AbjadImage.deskew_image(out)
    out = AbjadImage.binarize(out)
    out, dots_diacritics_mask = AbjadImage.remove_dots_diacritics(out)
    out = AbjadImage.seperate_connected_components(out, dots_diacritics_mask)
    out = AbjadImage.get_skeleton(out)
    _, _, to_predict_chars = AbjadImage.psp_segmentation(out)

    predicted_word = AbjadKeras.cnn_predict_word_as_chars(model, to_predict_chars, labels)
    predicted_word = predicted_word[::-1]

    print('PREDICTED WORD: ', predicted_word)
    with tf.get_default_graph().as_default():
        response = predicted_word
        return response

def main():
    global img_width, img_height, model, labels, fit_and_save

    labels = AbjadData.get_labels()
    model = AbjadKeras.cnn_model(img_width, img_height)
    model = AbjadKeras.load_weights(model)
    inp, outputs, functor = AbjadKeras.get_cnn_model_layers(model)

    if fit_and_save is True:
        (x_train, y_train), (x_test, y_test), abjad_labels = AbjadData.load_data(img_width, img_height)
        (x_train, y_train), (x_test, y_test) = AbjadData.prepare_data(x_train, y_train, x_test, y_test)
        model, history = AbjadKeras.cnn_fit(model, x_train, y_train, x_test, y_test, 50, 270)
        score = AbjadKeras.cnn_evaluate(model, x_test, y_test)
        AbjadKeras.save_weights(model)
        AbjadKeras.save_json(model)
    else:
        AbjadKeras.save_json(model)

    port = int(os.environ.get('PORT', '5000'))
    app.run(host = '0.0.0.0', port = port)

if __name__ == '__main__':
    main()
