# Cloned and modified from https://github.com/lunardog/convnets-keras
# Original author: https://github.com/heuritech
from keras import backend as K
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers import (Flatten, Dense, Dropout,  Activation, Input,
                          concatenate)
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.layer_utils import convert_all_kernels_in_model
import numpy as np
from scipy.misc import imread as sp_imread, imresize as sp_imresize


def preprocess_image_batch(image_paths, img_size=None, crop_size=None,
                           color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = sp_imread(im_path, mode='RGB')
        if img_size:
            img = sp_imresize(img, img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means
        # on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode == "bgr":
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[
                :,
                (img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2,
                (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2
            ]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                         ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch


def _crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        K.set_image_dim_ordering('th')
        if K.backend() == 'tensorflow':
            b, ch, r, c = X.get_shape()
        else:
            b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(
            K.permute_dimensions(square, (0, 2, 3, 1)),
            ((0, 0), (half, half)))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(n):
            if K.backend() == 'tensorflow':
                ch = int(ch)
            scale += alpha * extra_channels[:, i:i+ch, :, :]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def _splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        if K.backend() == 'tensorflow':
            div = int(X.get_shape()[axis]) // ratio_split
        else:
            div = X.shape[axis] // ratio_split

        if axis == 0:
            output = X[id_split*div:(id_split+1)*div, :, :, :]
        elif axis == 1:
            output = X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:, :, id_split*div:(id_split+1)*div, :]
        elif axis == 3:
            output = X[:, :, :, id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def AlexNet(weights=None, input_shape=(3, 227, 227)):
    K.set_image_dim_ordering('th')
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(96, (11, 11), strides=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = _crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = concatenate(
        [Convolution2D(128, (5, 5),
                       activation="relu",
                       name='conv_2_'+str(i+1))(
                           _splittensor(ratio_split=2, id_split=i)(conv_2)
                       )
         for i in range(2)],
        axis=1, name="conv_2"
    )

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = _crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, (3, 3), activation='relu',
                           name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = concatenate(
        [Convolution2D(192, (3, 3), activation="relu",
                       name='conv_4_'+str(i+1))(
                           _splittensor(ratio_split=2, id_split=i)(conv_4)
                       )
         for i in range(2)],
        axis=1, name="conv_4"
    )
    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = concatenate(
        [Convolution2D(128, (3, 3), activation="relu",
                       name='conv_5_'+str(i+1))(
                           _splittensor(ratio_split=2, id_split=i)(conv_5)
                       )
         for i in range(2)],
        axis=1, name="conv_5"
    )

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation("softmax", name="softmax")(dense_3)

    model = Model(inputs=inputs, outputs=prediction)

    if weights:
        model.load_weights(weights)

    if K.backend() == 'tensorflow':
        convert_all_kernels_in_model(model)

    model.compile(optimizer='nadam', loss='mse')

    return model
