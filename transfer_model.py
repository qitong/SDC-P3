from keras.models import model_from_json
import cv2
import numpy as np
from sklearn.utils import shuffle

def read_log_data():
    X_train_ = []
    y_train_ = []
    f = open('data/driving_log.csv')  #open('data/test.csv')
    for line in f.readlines():
        X_train_.append(line.split(',')[0].replace('/home/qitonghu/Desktop/simulator-linux/','data/'))
        y_train_.append(float(line.split(',')[3]))
    f.close()
    f = open('pullover/driving_log.csv')
    for line in f.readlines():
        X_train_.append(line.split(',')[0].replace('/home/qitonghu/Desktop/simulator-linux/pullover','pullover/'))
        y_train_.append(float(line.split(',')[3]))
    f.close()
    print(len(y_train_))
    return X_train_, y_train_

X_train_data, y_train_data = read_log_data()
X_train_data, y_train_data = shuffle(X_train_data, y_train_data)


def load_my_model(model_name):
    f = open(model_name+'.json')
    jsons = f.read()
    f.close()
    model = model_from_json(jsons)
    model.load_weights(model_name+'.h5')
    return model

original_model = load_my_model('model')

def get_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (80, 160))
    image = np.asarray(image, dtype=float)
    # image = np.array([np.dot(image[..., :3], [0.299, 0.587, 0.114]) / 255.0]).T
    image = (image / 255.) * 2 - 1.0
    return np.array(image)


def predict_images_with_model(images, model):
    for image_path in images:
        image = get_image(image_path)
        print(model.predict(image))


def freeze_part_model(model):
    trainable_flag = False
    for layer in model.layers:
        layer.trainable = trainable_flag
        if layer.name == 'flatten_1':
            trainable_flag = True

    model.compile(loss='mse', optimizer='adam')
    # model_conf = model.get_config()
    # for layer_conf in model_conf:
    #     print(layer_conf)
    return model

def transfer_learning(model):
    history = model.fit_generator(get_batch(), samples_per_epoch=len(y_train_data),
                                  nb_epoch=30)
    f = open('model_transfered.json', 'w')
    f.write(model.to_json())
    f.close()

    model.save_weights('model_transfered.h5')

def get_batch(batch_size=512):
    while 1:
        for i in range(len(y_train_data) // batch_size):
            start_index = i * batch_size
            end_index = i * batch_size + batch_size
            xx = np.array([get_image(x) for x in X_train_data[start_index:end_index]])
            yy = np.array(y_train_data[start_index:end_index])
            yield ({'convolution2d_input_1': xx}, {'dense_4': yy})

transfered_model = freeze_part_model(original_model)
transfer_learning(transfered_model)
