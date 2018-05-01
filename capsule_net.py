import os, math, csv
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from pandas import read_csv

def load_cifar_10():
    from keras.datasets import cifar10
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train,y_train),(x_test,y_test)

def load_cifar_100():
    from keras.datasets import cifar100
    num_classes = 100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train,y_train),(x_test,y_test)

def load_leaf():
    num_class = 99
    train_data = json.load(open("./train_features.json"))
    train_label = []
    train_feature = []
    for i in list(train_data.keys()):
        cur_pic = train_data[i]
        train_label.append(cur_pic['class_name'])
        train_feature.append(cv2.imread(os.path.join('./pad_images',i)))
    label_names = {}
    counter = 1
    for i in set(train_label):
        label_names[i] = counter
        counter += 1
    print(len(train_label))
    print(len(train_feature))
    data_count = len(train_feature)
    split = 0.8
    x_train = train_feature[:data_count*split]
    y_train = train_label[:data_count*split]
    x_test = train_feature[data_count*split:]
    y_test = train_label[data_count*split:]
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    x_train = np.asarray(x_train)
    x_test = np.asanyarray(x_test)
    print(label_names)
    return (x_train,y_train),(x_test,y_test)

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def initializer():
    if not os.path.exists('results/'):
        os.mkdir('results')
    if not os.path.exists('weights/'):
        os.mkdir('weights')

def plot_log(filename, show=True):
    # load data
    log_df = read_csv(filename)
    # epoch_list = [i for i in range(len(values[:,0]))]
    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for column in list(log_df):
        if 'loss' in column and 'val' in column:
            plt.plot(log_df['epoch'].tolist(),log_df[column].tolist(), label=column)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for column in list(log_df):
        if 'acc' in column :
            plt.plot(log_df['epoch'].tolist(),log_df[column].tolist(), label=column)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def data_generator(x,y,batch_size):
    x_train,y_train = x,y
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    generator = datagen.flow(x_train,y_train,batch_size=batch_size)
    while True:
        x,y  = generator.next()
        yield ([x,y],[y,x])
"""

Added minor tweak to allow user define clip value in Mask layer, other are same as XifengGuo repo code

Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset, 
not just on MNIST.
*NOTE*: some functions can be implemented in multiple ways, I keep all of them. You can try them for yourself just by
uncommenting them and commenting their counterparts.
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
    inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
    output: shape=[dim_1, ..., dim_{n-1}]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
        Mask Tensor layer by the max value in first axis
        Input shape: [None,d1,d2]
        Output shape: [None,d2]
    """
    clip_value = (0,1)

    def Mask(self,clip_value=(0,1),**kwargs):
        self.clip_value = clip_value # what if clip value is not 0 and 1?

    def call(self,inputs,**kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs,mask = inputs
        else:
            x = inputs
            # enlarge range of values in x by mapping max(new_x) = 1, others 
            x = (x - K.max(x,1,True)) / K.epsilon() + 1
            mask = K.clip(x,self.clip_value[0],self.clip_value[1]) # clip value beween 0 and 1
        masked_input = K.batch_dot(inputs, mask, [1,1])
        return masked_input

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None,input_shape[0][-1]])
    
        else:
            return tuple([None, input_shape[-1]])

def squash(vector, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vector), axis, keepdims=True)
    scale = s_squared_norm/(1+s_squared_norm)/K.sqrt(s_squared_norm)
    return scale*vector

class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
    [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self,input_shape):
        assert len(input_shape) >= 3, "Input tensor must have shape=[None, input_num_capsule,input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        self.W = self.add_weight(shape=[self.input_num_capsule,self.num_capsule,self.input_dim_vector,self.dim_vector],
                                initializer=self.kernel_initializer,
                                name='W')
        self.bias = self.add_weight(shape=[1,self.input_num_capsule,self.num_capsule,1,1],
                                initializer=self.bias_initializer,
                                name='bias',trainable=False)
        self.built = True

    def call(self,inputs,training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        """  
        # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
        # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
        w_tiled = K.tile(K.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])
        
        # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        """
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. This is faster but requires Tensorflow.
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
        """
        # Routing algorithm V1. Use tf.while_loop in a dynamic way.
        def body(i, b, outputs):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
            b = b + K.sum(inputs_hat * outputs, -1, keepdims=True)
            return [i-1, b, outputs]
        cond = lambda i, b, inputs_hat: i > 0
        loop_vars = [K.constant(self.num_routing), self.bias, K.sum(inputs_hat, 1, keepdims=True)]
        _, _, outputs = tf.while_loop(cond, body, loop_vars)
        """
        # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                # self.bias = K.update_add(self.bias, K.sum(inputs_hat * outputs, [0, -1], keepdims=True))
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
            # tf.summary.histogram('BigBee', self.bias)  # for debugging

        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])

def PrimaryCapsule(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)

from keras.layers import (
    Input,
    Conv2D,
    Activation,
    Dense,
    Flatten,
    Reshape,
    Dropout
)
from keras.layers.merge import add
from keras.regularizers import l2
from keras.models import Model
# from models.capsule_layers import CapsuleLayer, PrimaryCapsule, Length,Mask
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import optimizers
# from utils.helper_function import load_cifar_10,load_cifar_100
# from models.capsulenet import CapsNet as CapsNetv1
import numpy as np


def convolution_block(input,kernel_size=8,filters=16,kernel_regularizer=l2(1.e-4)):
    conv2 = Conv2D(filters=filters,kernel_size=kernel_size,kernel_regularizer=kernel_regularizer,
                    kernel_initializer="he_normal",padding="same")(input)
    norm = BatchNormalization(axis=3)(conv2)
    activation = Activation("relu")(norm)
    return activation    

def CapsNetv1(input_shape,n_class,n_route,n_prime_caps=32,dense_size = (512,1024)):
    conv_filter = 256
    n_kernel = 24
    primary_channel =64
    primary_vector = 9
    vector_dim = 9

    target_shape = input_shape

    input = Input(shape=input_shape)

    # TODO: try leaky relu next time
    conv1 = Conv2D(filters=conv_filter,kernel_size=n_kernel, strides=1, padding='valid', activation='relu',name='conv1',kernel_initializer="he_normal")(input)

    primary_cap = PrimaryCapsule(conv1,dim_vector=8, n_channels=64,kernel_size=9,strides=2,padding='valid')

    routing_layer = CapsuleLayer(num_capsule=n_class, dim_vector=vector_dim, num_routing=n_route,name='routing_layer')(primary_cap)

    output = Length(name='output')(routing_layer)

    y = Input(shape=(n_class,))
    masked = Mask()([routing_layer,y])
    
    x_recon = Dense(dense_size[0],activation='relu')(masked)

    for i in range(1,len(dense_size)):
        x_recon = Dense(dense_size[i],activation='relu')(x_recon)
    # Is there any other way to do  
    x_recon = Dense(target_shape[0]*target_shape[1]*target_shape[2],activation='relu')(x_recon)
    x_recon = Reshape(target_shape=target_shape,name='output_recon')(x_recon)

    return Model([input,y],[output,x_recon])

# why using 512, 1024 Maybe to mimic original 10M params?
def CapsNetv2(input_shape,n_class,n_route,n_prime_caps=32,dense_size = (512,1024)):
    conv_filter = 64
    n_kernel = 16
    primary_channel =64
    primary_vector = 12
    capsule_dim_size = 8

    target_shape = input_shape

    input = Input(shape=input_shape)

    # TODO: try leaky relu next time
    conv_block_1 = convolution_block(input,kernel_size=16,filters=64)
    primary_cap = PrimaryCapsule(conv_block_1,dim_vector=capsule_dim_size,n_channels=primary_channel,kernel_size=9,strides=2,padding='valid')    
    # Suppose this act like a max pooling 
    routing_layer = CapsuleLayer(num_capsule=n_class,dim_vector=capsule_dim_size*2,num_routing=n_route,name='routing_layer_1')(primary_cap)
    output = Length(name='output')(routing_layer)

    y = Input(shape=(n_class,))
    masked = Mask()([routing_layer,y])
    
    x_recon = Dense(dense_size[0],activation='relu')(masked)

    for i in range(1,len(dense_size)):
        x_recon = Dense(dense_size[i],activation='relu')(x_recon)
    # Is there any other way to do  
    x_recon = Dense(np.prod(target_shape),activation='relu')(x_recon)
    x_recon = Reshape(target_shape=target_shape,name='output_recon')(x_recon)

    # conv_block_2 = convolution_block(routing_layer)
    # b12_sum = add([conv_block_2,conv_block_1])

    return Model([input,y],[output,x_recon])

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(epochs=200,batch_size=64,mode=1):
    import numpy as np
    import os
    from keras import callbacks
    from keras.utils.vis_utils import plot_model
    if mode==1:
        num_classes = 10
        (x_train,y_train),(x_test,y_test) = load_cifar_10()
    else:
        num_classes = 99
        (x_train,y_train),(x_test,y_test) = load_leaf()
    model = CapsNetv1(input_shape=[32, 32, 3],
                        n_class=num_classes,
                        n_route=3)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model.summary()
    log = callbacks.CSVLogger('results/capsule-cifar-'+str(num_classes)+'-log.csv')
    tb = callbacks.TensorBoard(log_dir='results/tensorboard-capsule-cifar-'+str(num_classes)+'-logs',
                               batch_size=batch_size, histogram_freq=True)
    checkpoint = callbacks.ModelCheckpoint('weights/capsule-cifar-'+str(num_classes)+'weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    plot_model(model, to_file='models/capsule-cifar-'+str(num_classes)+'.png', show_shapes=True)

    model.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.1],
                  metrics={'output_recon':'accuracy','output':'accuracy'})

    generator = data_generator(x_train,y_train,batch_size)
    # Image generator significantly increase the accuracy and reduce validation loss
    model.load_weights('weights/capsule-cifar-10weights-45.h5') 
    model.fit_generator(generator,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        validation_data=([x_test, y_test], [y_test, x_test]),
                        epochs=epochs, verbose=1, max_q_size=100,
                        callbacks=[log,tb,checkpoint,lr_decay])

def test(epoch, mode=1):
    if mode == 1:
        num_classes =10
        _,(x_test,y_test) = load_cifar_10()
    else:
        num_classes = 100
        _,(x_test,y_test) = load_cifar_100()
    
    model = CapsNetv1(input_shape=[32, 32, 3],
                        n_class=num_classes,
                        n_route=3)
    model.load_weights('weights/capsule_weights/capsule-cifar-'+str(num_classes)+'weights-{:02d}.h5'.format(epoch)) 
    print("Weights loaded, start validation")   
    # model.load_weights('weights/capsule-weights-{:02d}.h5'.format(epoch))    
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-'*50)
    # Test acc: 0.7307
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save("results/real_and_recon.png")
    print('Reconstructed images are saved to ./results/real_and_recon.png')
    print('-'*50)

if __name__ == "__main__":
    train(150, 128, 1)
    # capsule_test(61)