import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img

root = './input'
np.random.seed(2016)
split_random_state = 7
split = .7
TRAIN = False
def load_numeric_training(standardize=True):
    data = pd.read_csv(os.path.join(root, 'train.csv'))
    ID = data.pop('id')
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(data).transform(data) if standardize else data.values
    return ID, X, y


def load_numeric_test(standardize=True):
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    ID = test.pop('id')
    test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return ID, test


def resize_img(img, max_dim=96):
    max_ax = max((0, 1), key=lambda i: img.size[i])
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, max_dim=96, center=True):
    X = np.empty((len(ids), max_dim, max_dim, 1))
    for i, idee in enumerate(ids):
        x = resize_img(load_img(os.path.join(root, 'images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        length = x.shape[0]
        width = x.shape[1]
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        X[i, h1:h2, w1:w2, 0:1] = x
    return np.around(X / 255.0)


def load_train_data(split=split, random_state=None):
    ID, X_num_tr, y = load_numeric_training()
    X_img_tr = load_image_data(ID)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(X_num_tr, y))
    X_num_val, X_img_val, y_val = X_num_tr[test_ind], X_img_tr[test_ind], y[test_ind]
    X_num_tr, X_img_tr, y_tr = X_num_tr[train_ind], X_img_tr[train_ind], y[train_ind]
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val)


def load_test_data():
    ID, X_num_te = load_numeric_test()
    X_img_te = load_image_data(ID)
    return ID, X_num_te, X_img_te

print('Loading the training data...')
(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = load_train_data(random_state=split_random_state)
y_tr_cat = to_categorical(y_tr)
y_val_cat = to_categorical(y_val)

from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img

class ImageDataGenerator2(ImageDataGenerator):
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        with self.lock:
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y

print('Creating Data Augmenter...')
# Data Augmentor
imgen = ImageDataGenerator2(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest')
imgen_train = imgen.flow(X_img_tr, y_tr_cat, seed=np.random.randint(1, 10000))
print('Finished making data augmenter...')

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Reshape
from keras import regularizers, optimizers
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# Customized Huber Loss used for reconstruction and prevent the loss from exposion
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = keras.backend.abs(error) < clip_delta
  squared_loss = keras.backend.square(error*10) * keras.backend.square(error*10)
  linear_loss  = clip_delta * (keras.backend.abs(error) - 0.5 * clip_delta)
  return tf.where(cond, squared_loss, linear_loss)

def combined_model_two_layer():
    image = Input(shape=(96, 96, 1), name='image')
    x = Convolution2D(8, 5, 5, input_shape=(96, 96, 1), border_mode='same')(image)
    x = (Activation('relu'))(x)
    x_1 = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = (Convolution2D(32, 7, 7, border_mode='same'))(x_1)
    x = (Activation('relu'))(x)
    x_2 = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    numerical = Input(shape=(192,), name='numerical')
    x_1d = Flatten()(x_1)
    x_2d = Flatten()(x_2)
    concatenated_d = merge([x_1d,x_2d, numerical], mode='concat')
    x = Dense(100, activation='relu')(concatenated_d)
    x = Dropout(.5)(x)
    recon = Dense(512,activation='relu')(x)
    recon = Dense(96*96*1,activation='relu')(recon)
    recon = Reshape(target_shape=(96,96,1),name='output_recon')(recon)
    out = Dense(99, activation='softmax')(x)
    model = Model(input=[image, numerical], output=[out])
    model.summary()
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
    return model

def combined_model():
    image = Input(shape=(96, 96, 1), name='image')
    x = Convolution2D(8, 5, 5, input_shape=(96, 96, 1), border_mode='same')(image)
    x = (Activation('relu'))(x)
    x_1 = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = (Convolution2D(32, 7, 7, border_mode='same'))(x_1)
    x = (Activation('relu'))(x)
    x_2 = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    x = (Convolution2D(64, 5, 5, border_mode='same'))(x_2)
    x = (Activation('relu'))(x)
    x_3 = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    numerical = Input(shape=(192,), name='numerical')
    x_1d = Flatten()(x_1)
    x_2d = Flatten()(x_2)
    x_3d = Flatten()(x_3)
    concatenated_d = merge([x_1d,x_2d,x_3d, numerical], mode='concat')
    x = Dense(100, activation='relu')(concatenated_d)
    x = Dropout(.5)(x)
    recon = Dense(512,activation='relu')(x)
    recon = Dense(96*96*1,activation='relu')(recon)
    recon = Reshape(target_shape=(96,96,1),name='output_recon')(recon)
    out = Dense(99, activation='softmax')(x)
    model = Model(input=[image, numerical], output=[out,recon])
    model.summary()
    model.compile(loss=['categorical_crossentropy','mse'], optimizer='adam', metrics=['accuracy'])
    return model

print('Creating the model...')
model = combined_model_two_layer()
print('Model created!')



def combined_generator(imgen, X):
    while True:
        for i in range(X.shape[0]):
            batch_img, batch_y = next(imgen)
            x = X[imgen.index_array]
            yield [batch_img, x], [batch_y]

best_model_file = "leafnet_good_two_recon.h5"

if (TRAIN):
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
    model = combined_model_two_layer()
    # model = load_model(best_model_file)
    print('Training model...')
    history = model.fit_generator(combined_generator(imgen_train, X_num_tr),
                                samples_per_epoch=64,
                                nb_epoch=100000,
                                validation_data=([X_img_val, X_num_val], [y_val_cat]),
                                nb_val_samples=X_num_val.shape[0],
                                verbose=0,
                                callbacks=[best_model])

print('Loading the best model...')
model = load_model(best_model_file,custom_objects={'huber_loss': huber_loss})
print('Best Model loaded!')
# _, predictions_valid = model.predict([X_img_val,X_num_val])
predictions_valid= model.predict([X_img_val,X_num_val])
from sklearn.metrics import log_loss
score = log_loss(y_val_cat, predictions_valid)
print('=====')
print(score)
print('+++++')
LABELS = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())
index, test, X_img_te = load_test_data()
yPred_proba = model.predict([X_img_te, test])
yPred = pd.DataFrame(yPred_proba,index=index,columns=LABELS)
print('Creating and writing submission...')
fp = open('submit.csv', 'w')
fp.write(yPred.to_csv())
print('Finished writing submission')