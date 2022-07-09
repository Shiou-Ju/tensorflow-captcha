# Learnt from Andy Wu's Note
# https://notes.andywu.tw/2019/%E7%94%A8tensorflowkeras%E8%A8%93%E7%B7%B4%E8%BE%A8%E8%AD%98%E9%A9%97%E8%AD%89%E7%A2%BC%E7%9A%84cnn%E6%A8%A1%E5%9E%8B/

import sys
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


img_rows = None
img_cols = None
digits_in_img = 4
model = None
np.set_printoptions(suppress=True, linewidth=150, precision=9,
                    formatter={'float': '{: 0.9f}'.format})


# # Mapping characters to integers
# char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# # Mapping integers back to original characters
# num_to_char = layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
# )


alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"


def convert_indices_to_alpha_numeric(data):
    #Creates a dict, that maps to every char of alphabet an unique int based on position
    # char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    encoded_data = []
    char_to_int = list(alphabet)
    #Replaces every char in data with the mapped int
    # encoded_data.append([char_to_int[char] for char in data])
    print(char_to_int)
    # temp = [list(char_to_int.keys()).index(i) for i in data]
    temp = [char_to_int[i] for i in data]
    # print(temp)
    # encoded_data.append()
    print(temp)  # Prints the int encoded array
    return temp



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

img_filename = input('Varification code img filename: ')
img = load_img(img_filename, color_mode='grayscale')
img_array = img_to_array(img)
img_rows, img_cols, _ = img_array.shape
x_list = split_digits_in_img(img_array)

# label_encoder = LabelEncoder()

varification_code = list()
for i in range(digits_in_img):
    confidences = model.predict(np.array([x_list[i]]), verbose=0)
    # result_class = model.predict_classes(np.array([x_list[i]]), verbose=0)
    # The line above does not work in tensorflow 2.7, see
    # https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes
    predict_x = model.predict(np.array([x_list[i]]), verbose=0)
    result_class = np.argmax(predict_x, axis=1)
    # print(result_class)
    # converted = convert_indices_to_alpha_numeric(result_class[0])
    varification_code.append(result_class[0])
    # pridct_print = label_encoder.inverse_transform(varification_code)
    # print(pridct_print)
    print('Digit {0}: Confidence=> {1}    Predict=> {2}'.format(
        i + 1, np.squeeze(confidences), np.squeeze(result_class)))

converted = convert_indices_to_alpha_numeric(varification_code)
print('Predicted varification code:', converted)
