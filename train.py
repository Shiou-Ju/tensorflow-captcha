# Learnt from Andy Wu's Note
# https://notes.andywu.tw/2019/%E7%94%A8tensorflowkeras%E8%A8%93%E7%B7%B4%E8%BE%A8%E8%AD%98%E9%A9%97%E8%AD%89%E7%A2%BC%E7%9A%84cnn%E6%A8%A1%E5%9E%8B/
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


epochs = 100  # 訓練的次數
img_rows = None  # 驗證碼影像檔的高
img_cols = None  # 驗證碼影像檔的寬
# TODO: migrate to English letters & numbers
digits_in_img = 4  # 驗證碼影像檔中有幾位數，包含英文小寫
x_list = list()  # 存所有驗證碼內容影像檔的array
y_list = list()  # 存所有的驗證碼內容影像檔array代表的正確內容
x_train = list()  # 存訓練用驗證碼內容影像檔的array
y_train = list()  # 存訓練用驗證碼內容影像檔array代表的正確內容
x_test = list()  # 存測試用驗證碼內容影像檔的array
y_test = list()  # 存測試用驗證碼內容影像檔array代表的正確內容

# # Mapping characters to integers
# char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# # Mapping integers back to original characters
# num_to_char = layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
# )

# Is the alphabet of all possible chars you want to convert
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"


def convert_alpha_numeric_to_indices(data):
    #Creates a dict, that maps to every char of alphabet an unique int based on position
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    encoded_data = []
    #Replaces every char in data with the mapped int
    # encoded_data.append([char_to_int[char] for char in data])
    temp = [char_to_int[char] for char in data]
    # print(temp)
    # encoded_data.append()
    print(temp)  # Prints the int encoded array
    return temp

    #This part now replaces the int by an one-hot array with size alphabet
    # one_hot = []
    # for value in encoded_data:
    #     #At first, the whole array is initialized with 0
    #     letter = [0 for _ in range(len(alphabet))]
    #     #Only at the number of the int, 1 is written
    #     letter[value] = 1
    #     one_hot.append(letter)
    # return one_hot



def split_digits_in_img(img_array, x_list, y_list):
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(img_filename[i])
        print(y_list)
        print(img_filename)


img_filenames = os.listdir('training')

for img_filename in img_filenames:
    # TODO:
    # if '.png' not in img_filename:
    if '.jpeg' not in img_filename:
        continue
    img = load_img('training/{0}'.format(img_filename), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, x_list, y_list)


# 1-1. ValueError: invalid literal for int() with base 10: 'a'
# y_list = keras.utils.to_categorical(y_list, num_classes=36)

# label_encoder = LabelEncoder()
# vec = label_encoder.fit_transform(y_list)

encoded_data = convert_alpha_numeric_to_indices(y_list)

# 1-2. ValueError: Shapes (None, 36) and (None, 10) are incompatible
# y_list = keras.utils.to_categorical(vec, num_classes=36)


# 1-3. IndexError: index 10 is out of bounds for axis 1 with size 10
# y_list = keras.utils.to_categorical(vec, num_classes=10)

y_list = to_categorical(encoded_data, num_classes=36)

x_train, x_test, y_train, y_test = train_test_split(x_list, y_list)


if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
    print('Model loaded from file.')
else:
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
              input_shape=(img_rows, img_cols // digits_in_img, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    # Modify as per 1-2
    # Success, low accuracy => val_accuracy: 0.0741
    # TODO:
    # But wrong out put: Predicted varification code: [10, 11, 4, 8] (ugsi)
    # model.add(layers.Dense(10, activation='softmax'))
    model.add(layers.Dense(36, activation='softmax'))
    print('New model created.')

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(), metrics=['accuracy'])


model.fit(np.array(x_train), np.array(y_train), batch_size=digits_in_img,
          epochs=epochs, verbose=1, validation_data=(np.array(x_test), np.array(y_test)))

loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

model.save('cnn_model.h5')
