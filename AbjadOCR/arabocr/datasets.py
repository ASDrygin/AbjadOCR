import numpy as np
from PIL import Image
import keras
import cv2
import glob
from arabocr.preprocessing import AbjadImage
import os

class AbjadData(object):
   
    #   TODO : Upload dataset and download it on call, for now it's LOCAL
    @classmethod
    def load_data(cls, img_width, img_height, download = False):
        print('')
        
        (x_train, y_train), (x_test, y_test) = ([], []), ([], [])
        abjad_labels = np.array(['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'])

        if download == False and os.path.exists('x_train.txt') and os.path.exists('x_test.txt'):
            print('* Loading abjad dataset from local:')

            x_train = np.loadtxt("x_train.txt", np.float32)
            y_train = np.loadtxt("y_train.txt", np.float32)

            x_test = np.loadtxt("x_test.txt", np.float32)
            y_test = np.loadtxt("y_test.txt", np.float32)

            x_train = x_train.reshape((-1, img_width, img_height, 1)).astype('float32')
            x_test = x_test.reshape((-1, img_width, img_height, 1)).astype('float32')
            
        else:
            print('* Loading abjad dataset from host:')

            for lab_idx, lab_value in enumerate(abjad_labels):
                train_lab_dir = 'train/' + lab_value + '/*.png'
                print(' [', lab_value, ']')

                train_samples_list = sorted(glob.glob(train_lab_dir)) 

                for train_sample in train_samples_list:
                    curr_img = np.array(Image.open(train_sample))
                    curr_img = AbjadImage.invert_binary(curr_img)

                    curr_resized_img = cv2.resize(curr_img, (img_width, img_height))

                    x_train = np.append(x_train, curr_resized_img)
                    y_train = np.append(y_train, lab_idx)

                test_lab_dir = 'test/' + lab_value + '/*.png'
                test_samples_list = sorted(glob.glob(test_lab_dir)) 

                for test_sample in test_samples_list:
                    curr_img = np.array(Image.open(test_sample))
                    curr_img = AbjadImage.invert_binary(curr_img)

                    curr_resized_img = cv2.resize(curr_img, (img_width, img_height))

                    x_test = np.append(x_test, curr_resized_img)
                    y_test = np.append(y_test, lab_idx)

            x_train = x_train.reshape((-1, img_width, img_height, 1)).astype('float32')
            x_test = x_test.reshape((-1, img_width, img_height, 1)).astype('float32')

            x_train_to_save = x_train.reshape((-1, img_width * img_height)).astype('float32')
            x_test_to_save = x_test.reshape((-1, img_width * img_height)).astype('float32')

            np.savetxt("x_train.txt", x_train_to_save)         
            np.savetxt("x_test.txt", x_test_to_save)
            
            np.savetxt("y_train.txt", y_train)         
            np.savetxt("y_test.txt", y_test) 

        print('')
        return (x_train, y_train), (x_test, y_test), abjad_labels

    @classmethod
    def prepare_data(cls, x_train, y_train, x_test, y_test):
        num_classes = 28

        # normalize input from 0-255 to 0-1
        x_train = x_train / 255
        x_test = x_test / 255

        # one-hot encode outputs data
        y_train = keras.utils.to_categorical(y_train, num_classes = num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes = num_classes)

        return (x_train, y_train), (x_test, y_test)

    @classmethod
    def get_labels(cls):
        abjad_labels = np.array(['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'])
        return abjad_labels
