import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
import cv2
import numpy as np
from sklearn.utils import shuffle


def read_log_data():
    f = open('data/driving_log.csv')  #open('data/test.csv')
    X_train_ = []
    y_train_ = []
    for line in f.readlines():
        X_train_.append(line.split(',')[0].replace('/home/qitonghu/Desktop/simulator-linux/',''))
        y_train_.append(float(line.split(',')[3]))
    return X_train_, y_train_

X_train_data, y_train_data = read_log_data()
X_train_data, y_train_data = shuffle(X_train_data, y_train_data)
# X_train_data = X_train_data[:1000]
# y_train_data = y_train_data[:1000]


def get_image(file_path):
    image = cv2.imread('data/' + file_path)
    image = cv2.resize(image, (80, 160))
    image = np.asarray(image, dtype=float)
    # image = np.array([np.dot(image[..., :3], [0.299, 0.587, 0.114]) / 255.0]).T
    image = (image / 255.) * 2 - 1.0
    return np.array(image)

def get_batch(batch_size=128):
    while 1:
        for i in range(len(y_train_data) // batch_size):
            start_index = i * batch_size
            end_index = i * batch_size + batch_size
            xx = np.array([get_image(x) for x in X_train_data[start_index:end_index]])
            yy = np.array(y_train_data[start_index:end_index])
            yield ({'convolution2d_input_1': xx}, {'dense_4': yy})

def build_model():
    model = Sequential()
    # Normalization

    # Conv
    nb_kernels = [24, 36, 48, 64, 72, 84]
    kernel_sizes = [5, 5, 5, 3, 3, 2]
    kernel_strides = [2, 2, 2, 1, 1, 1]

    # 3@160x80 -> 24@78x38 -> 36@37x17 -> 48@16x6 -> 64@14x4 -> 72@12x2 -> 84@11x1 -> 924 -> 100 -> 50 -> 10 -> 1
    model.add(Convolution2D(nb_kernels[0], kernel_sizes[0], kernel_sizes[0],
                             subsample=(kernel_strides[0], kernel_strides[0]), activation='relu', input_shape=(160, 80, 3)))

    for i in range(1, len(nb_kernels)):
        model.add(Convolution2D(nb_kernels[i], kernel_sizes[i], kernel_sizes[i]))
        model.add(MaxPooling2D((kernel_strides[i], kernel_strides[i])))
        #model.add(Dropout(0.5))
        model.add(Activation('tanh'))

    # FC
    model.add(Flatten())
    nb_neurons = [100, 50, 10]
    for i in range(len(nb_neurons)):
        model.add(Dense(nb_neurons[i]))
        #model.add(Dropout(0.5))
        model.add(Activation('tanh'))

    # output
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model


def train_model(model):
    history = model.fit_generator(get_batch(), samples_per_epoch=len(y_train_data)*3,
                                  nb_epoch=20)
    f = open('model.json', 'w')
    f.write(model.to_json())
    f.close()

    model.save_weights('model.h5')

model = build_model()
train_model(model)
