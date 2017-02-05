from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import h5py
import json
from keras import backend as K
from sklearn.utils import shuffle
K.set_image_dim_ordering('th')

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, 224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    # if weights_path:
    #     model.load_weights(weights_path)

    # assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    for layer in model.layers:
        layer.trainable = False

    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Dense(2048, activation='relu'))
    # top_model.add(Dropout(0.5))
    # top_model.add(Dense(128, activation='relu'))
    # top_model.add(Dropout(0.5))
    # top_model.add(Dense(1))
    #
    # model.add(top_model)



    #model.compile(loss='mse', optimizer='adam')

    return model


def train_model(model):
    history = model.fit_generator(get_batch(), samples_per_epoch=len(y_train_data),
                                  nb_epoch=15)
    f = open('model.json', 'w')
    f.write(model.to_json())
    f.close()

    model.save_weights('model.h5')


def predict_with_model(data_line, model):
    result = model.predict(get_image(data_line))
    return result

def read_log_data():
    f = open('data/driving_log.csv')  #open('data/test.csv')
    X_train_ = []
    y_train_ = []

    for line in f.readlines():
        X_train_.append(line.split(',')[0].replace('/home/qitonghu/Desktop/simulator-linux/',''))
        y_train_.append(float(line.split(',')[3]))
    f.close()
    f = open('pullover/driving_log.csv')
    for line in f.readlines():
        X_train_.append(line.split(',')[0])
        y_train_.append(float(line.split(',')[3]))
    f.close()
    return X_train_, y_train_

X_train_data, y_train_data = read_log_data()
#X_train_data, y_train_data = shuffle(X_train_data, y_train_data)
X_train_data = X_train_data[:10]
y_train_data = y_train_data[:10]


def get_image(file_path):
    image = cv2.imread('data/' + file_path)
    image = cv2.resize(image, (224, 224))
    image = np.asarray(image, dtype=float)
    image = image.transpose((2, 0, 1))
    image = (image / 255.) * 2 - 1.0
    return np.array([image])

def train_model(model):
    history = model.fit_generator(get_batch(), samples_per_epoch=len(y_train_data),
                                      nb_epoch=10)
    f = open('model_vgg.json', 'w')
    f.write(model.to_json())
    f.close()

    model.save_weights('model_vgg.h5')

def get_batch(batch_size=512):
    while 1:
        for i in range(len(y_train_data) // batch_size):
            start_index = i * batch_size
            end_index = i * batch_size + batch_size
            xx = np.array([get_image(x) for x in X_train_data[start_index:end_index]])
            yy = np.array(y_train_data[start_index:end_index])
            yield ({'zeropadding2d_input_1': xx}, {'sequential_2': yy})

model = VGG_16('vgg16_weights.h5')

#train_model(model)
new_X_train = []
new_y_train = []
count = 0
for i in range(len(X_train_data)):
    count += 1
    if count % 1000 == 0:
        print(count)
    result = predict_with_model(X_train_data[i], model)[0]
    target = y_train_data[i]
    new_X_train.append(result.tolist())
    new_y_train.append(target)
f = open('extracted_feature', 'w')
f.write(json.dumps(new_X_train))
f.close()
f = open('target','w')
f.write(json.dumps(new_y_train))
f.close()

                # if __name__ == "__main__":
#     im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#     im[:,:,0] -= 103.939
#     im[:,:,1] -= 116.779
#     im[:,:,2] -= 123.68
#     im = im.transpose((2,0,1))
#     im = np.expand_dims(im, axis=0)
#
#     # Test pretrained model
#     model = VGG_16('vgg16_weights.h5')
#     print('sadsadasuidhi')
#     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(optimizer=sgd, loss='categorical_crossentropy')
#     out = model.predict(im)
#     print(np.argmax(out))