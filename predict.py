# Learnt from Andy Wu's Note
# Adjusted to actual needs of analyzing images with English letters and numbers


import os
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


img_rows = None
img_cols = None
digits_in_img = 4
model = None
np.set_printoptions(suppress=True, linewidth=150, precision=9,
                    formatter={'float': '{: 0.9f}'.format})


alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"


def convert_indices_to_alpha_numeric(data):
    char_to_int = list(alphabet)
    print(char_to_int)
    encoded_data = [char_to_int[i] for i in data]
    print(encoded_data)  # Prints the int encoded array
    return encoded_data


def split_digits_in_img(img_array):
    x_list = list()
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
    return x_list


if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
else:
    print('No trained model found.')
    exit(-1)

# TODO: add batch operation of a folder
img_filename = input('Varification code img filename: ')
img = load_img(img_filename, color_mode='grayscale')
img_array = img_to_array(img)
img_rows, img_cols, _ = img_array.shape
x_list = split_digits_in_img(img_array)


varification_code = list()
for i in range(digits_in_img):
    confidences = model.predict(np.array([x_list[i]]), verbose=0)
    # result_class = model.predict_classes(np.array([x_list[i]]), verbose=0)
    # The line above does not work in tensorflow 2.7, see
    # https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes
    predict_x = model.predict(np.array([x_list[i]]), verbose=0)
    result_class = np.argmax(predict_x, axis=1)
    varification_code.append(result_class[0])
    print('Digit {0}: Confidence=> {1}    Predict=> {2}'.format(
        i + 1, np.squeeze(confidences), np.squeeze(result_class)))

converted = convert_indices_to_alpha_numeric(varification_code)
print('Predicted varification code:', converted)
