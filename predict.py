# Inspired by Andy Wu's Note, please see README.
# Adjusted to actual needs of analyzing images with English letters and numbers


import os
import argparse
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

img_rows = None
img_cols = None
digits_in_img = 4
model = None
verbose_print_each_char = False
correct_prediction = 0
total_file_counts = 0


# parse args
parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("-v ", "--verbose_print",
                    help="Print out prediction char by char.", action="store_true")

args = parser.parse_args()

if(args.verbose_print == True):
    verbose_print_each_char = True


np.set_printoptions(suppress=True, linewidth=150, precision=9,
                    formatter={'float': '{: 0.9f}'.format})

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"


def convert_indices_to_alpha_numeric(data):
    char_to_int = list(alphabet)
    encoded_data = [char_to_int[i] for i in data]
    return encoded_data


def split_digits_in_img(img_array, original_img_filename):
    x_list = list()
    y_list = list()
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        # answer splitted from filename
        y_list.append(original_img_filename[i])
    return x_list, y_list


if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
else:
    print('No trained model found.')
    exit(-1)

validation_folder = 'validation'
img_filenames = os.listdir(validation_folder)

for original_img_filename in img_filenames:
    img_filename = os.path.join(validation_folder, original_img_filename)
    img = load_img(img_filename, color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    x_list, y_list = split_digits_in_img(img_array, original_img_filename)

    varification_code = list()
    for i in range(digits_in_img):
        confidences = model.predict(np.array([x_list[i]]), verbose=0)
        # result_class = model.predict_classes(np.array([x_list[i]]), verbose=0)
        # The line above does not work in tensorflow 2.7, see
        # https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes
        predict_x = model.predict(np.array([x_list[i]]), verbose=0)
        result_class = np.argmax(predict_x, axis=1)
        varification_code.append(result_class[0])
        if verbose_print_each_char:
            print('Digit {0}: Confidence=> {1}    Predict=> {2}'.format(
                i + 1, np.squeeze(confidences), np.squeeze(result_class)))

    converted = convert_indices_to_alpha_numeric(varification_code)
    print('Predicted content:', converted)
    print('Answer:           ', y_list)
    print(converted == y_list)
    if (converted == y_list):
        correct_prediction += 1
    total_file_counts += 1
    print('#######################################################')

print('Correct %: ', correct_prediction/total_file_counts)
