# Learnt from Andy Wu's Note
# https://notes.andywu.tw/2019/%E7%94%A8tensorflowkeras%E8%A8%93%E7%B7%B4%E8%BE%A8%E8%AD%98%E9%A9%97%E8%AD%89%E7%A2%BC%E7%9A%84cnn%E6%A8%A1%E5%9E%8B/


import numpy as np
import os
import sys
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


img_rows = None
img_cols = None
digits_in_img = 6
model = None
np.set_printoptions(suppress=True, linewidth=150, precision=9,
                    formatter={'float': '{: 0.9f}'.format})


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
print('Predicted varification code:', varification_code)
