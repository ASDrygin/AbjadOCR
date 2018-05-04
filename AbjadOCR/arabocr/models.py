import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
import cv2
import os

class AbjadKeras(object):
    @classmethod
    def cnn_model(cls, img_width, img_height, num_classes = 28):
        model = Sequential()

        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu', input_shape = (img_width,img_height,1)))
        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation = "softmax"))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = [ "accuracy" ])
        
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

        return model

    @classmethod
    def get_cnn_model_layers(cls, model):
        inp = model.input                                           # input placeholder
        outputs = [layer.output for layer in model.layers]          # all layer outputs
        functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

        return inp, outputs, functor

    @classmethod
    def cnn_predict_word_as_chars(cls, model, to_predict_chars, labels, img_width = 32, img_height = 32):
        i = 0
        res = ""

        while i < len(to_predict_chars):
            curr_img = to_predict_chars[i]

            curr_img = curr_img.reshape((-1, img_width, img_height, 1)).astype('float32')
            res += AbjadKeras.get_prediction(model, curr_img, labels)
            i += 1

        return res

    @classmethod
    def get_prediction(cls, model, img_data, labels):
        prediction = model.predict(img_data)[0]
        
        if (np.isnan(prediction[0])):
            return 'NaN'

        best_class = ''
        best_conf = 0
    
        for lab_idx, lab_value in enumerate(labels):
            if(prediction[lab_idx] > best_conf):
                best_class = lab_idx
                best_conf = prediction[lab_idx]

        with tf.get_default_graph().as_default():
            response = labels[best_class]
            return response

    @classmethod
    def get_prediction_confidences(cls, model, img_data, labels):
        prediction = model.predict(img_data)[0]

        if (np.isnan(prediction[0])):
            return 'NaN'

        dict_confz = {labels[i] : prediction[i] for i in range(0, len(prediction), 1)}
        dict_confz = sorted(dict_confz.items(), key=lambda x: x[1])

        return dict_confz

    @classmethod
    def cnn_fit(cls, model,x_train, y_train, x_test = [], y_test = [], epochs = 27, batch_size = 300):
        learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            #rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1,  # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(x_train)

        history = None

        if(x_test != [] and y_test != []):
            history = model.fit_generator(datagen.flow(x_train, y_train,
                                      batch_size=batch_size),
                                      epochs = epochs,
                                      validation_data = (x_test, y_test),
                                      verbose = 2,
                                      steps_per_epoch= batch_size,
                                      callbacks=[learning_rate_reduction])
        else:
            history = model.fit_generator(datagen.flow(x_train, y_train,
                                      batch_size=batch_size),
                                      epochs = epochs,
                                      verbose = 2,
                                      steps_per_epoch= batch_size,
                                      callbacks=[learning_rate_reduction])

        return model, history

    @classmethod
    def cnn_evaluate(cls, model, x_test, y_test):
        scores = model.evaluate(x_test, y_test, verbose = 0)
        print("Accuray: %.2f%%" % (scores[1] * 100))

        return scores[1] * 100
    
    @classmethod
    def save_weights(cls, model):
        model.save_weights('abjad_weights.h5')

        return 'Abjad weights saved.'

    @classmethod
    def load_weights(cls, model):
        model.load_weights('abjad_weights.h5')

        return model

    @classmethod
    def save_json(cls, model):
        with open('model_architecture.json', 'w') as f:
            f.write(model.to_json())
        return 'Saved to JSON.'

    @classmethod
    def load_json(cls, model):
        with open('model_architecture.json', 'r') as f:
            model = model_from_json(f.read())

        return model