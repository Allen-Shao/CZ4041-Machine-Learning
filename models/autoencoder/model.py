from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.callbacks import TensorBoard
import os
import pandas as pd
from scipy.misc import imread, imresize
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Merge
from keras.layers import Input, Dense
from keras.models import model_from_json, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from ae import get_encoder

def prepare_data(data_dir, img_height, img_width):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    image_data = {}
    images_dir = os.path.join(data_dir, 'images')
    for image_file in os.listdir(images_dir):
        image_data[image_file.split(".")[0]] = imresize(imread(os.path.join(images_dir, image_file)),(img_height,img_width)).reshape((img_height, img_width,1)).astype(np.float32)
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)
    labels_cat = to_categorical(labels)
    classes = list(le.classes_)

    test_ids = test.id
    train_ids = train.id
    image_train = np.array([image_data[str(_)] for _ in train_ids])
    image_test = np.array([image_data[str(_)] for _ in test_ids])

    train = train.drop(['id', 'species'], axis=1)
    test = test.drop(['id'], axis=1)

    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, labels_cat, classes, test_ids, test, np.concatenate((image_train, image_test)) 

def encode_images(images, data_dir):
    input_img = Input(shape=images[0].shape)
    images = (np.random.random(images.shape) < images).astype(np.float32)
    encoder = get_encoder(input_img, images, data_dir)
    return encoder.predict(images, batch_size=128)

def construct_model(data_dir):
    model = Sequential()
    model.add(Dense(1024, input_dim=448, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(99, kernel_initializer="glorot_normal"))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    if os.path.exists(os.path.join(data_dir, 'model_weights.h5')):
        model.load_weights(os.path.join(data_dir, 'model_weights.h5'))
    	return model, True
    return model, False

def train_and_predict(train_data, test_data, labels, images, data_dir, epochs):
    encoded_images = encode_images(images, data_dir) 
    encoded_images = encoded_images.reshape(encoded_images.shape[0], -1)
    train_data = np.concatenate((train_data, encoded_images[:train_data.shape[0],:]), axis=1)
    test_data = np.concatenate((test_data, encoded_images[train_data.shape[0]:,:]), axis=1)

    model, trained = construct_model(data_dir)
    if (not trained):
        checkpointFilePath = os.path.join("dense_weights/","weights-improvement-{epoch:02d}-{loss:.8f}.hdf5")
        checkpoint = ModelCheckpoint(checkpointFilePath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        model.fit(train_data, labels, epochs=epochs, batch_size=256, callbacks=[checkpoint, TensorBoard(log_dir='/tmp/dense')])
        model.save_weights(os.path.join(data_dir, 'model_weights.h5'))
    pred = model.predict_proba(test_data)
    return pred

def submit(preds, test_ids, classes):
    submission = pd.DataFrame(preds, columns=classes)
    submission.insert(0, 'id', test_ids)
    submission.reset_index()
    submission.to_csv('submission.csv', index=False)

def main():
    img_height, img_width = 128, 128
    data_dir = 'data/' 
    epochs = 5000
    train, labels, classes, test_ids, test, images = prepare_data(data_dir, img_height, img_width)
    pred = train_and_predict(train, test, labels, images, data_dir, epochs)
    submit(pred, test_ids, classes)

if __name__ == '__main__':
    main()

