# -*- coding: utf-8 -*-
'''
3d-resnet
'''

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution3D, MaxPooling3D, ZeroPadding3D, \
    AveragePooling2D, TimeDistributed, Dropout

from keras.layers.merge import Concatenate

from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization


def get_weight_path():
    if K.image_data_format() == 'th':
        return '3dnet_weights_th_dim_ordering_th_kernels_notop.h5'
    else:
        return '3dnet_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height, dense):
    def get_output_length(input_length):
        input_length = input_length // 2
        return input_length
    return get_output_length(width), get_output_length(height), get_output_length(dense)


def nn_base(input_tensor=None, trainable=False):
    # Determine proper input shape
    #input_shape=(image_size, image_size, image_size, num_channels=1) # l, h, w, c
    if K.image_data_format() == 'th':
        input_shape = (1, None, None, None)
    else:
        input_shape = (None, None, None, 1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'tf':
        bn_axis = 4
    else:
        bn_axis = 1

    m = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(img_input)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    shortcut = Convolution3D(16, (3, 3, 3), activation='relu', padding='same')(m)   
    #Bottleneck
    mNew = Convolution3D(16, (1, 1, 1), activation='relu', padding='same')(m)
    mNew = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(mNew)
    mNew = Convolution3D(16, (1, 1, 1), padding='same')(mNew)
    mNew = Dropout(0.7)(mNew)
    #mMerge = merge([shortcut, mNew], mode='concat', concat_axis=-1)
    mMerge = Concatenate(axis=-1)([shortcut, mNew])

    m = Activation('relu')(mMerge)

    shortcut2 = Convolution3D(16, (3, 3, 3), activation='relu', padding='same')(m)
    #Bottleneck
    mNew2 = Convolution3D(16, (1, 1, 1), activation='relu', padding='same')(m)
    mNew2 = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(mNew2)
    mNew2 = Convolution3D(16, (1, 1, 1), padding='same')(mNew2)
    mNew2 = Dropout(0.7)(mNew2)
    #mMerge2 = merge([shortcut2, mNew2], mode='concat', concat_axis=-1)
    mMerge2 = Concatenate(axis=-1)([shortcut2, mNew2])
    
    m = Activation('relu')(mMerge2)

    shortcut3 = Convolution3D(16, (3, 3, 3), activation='relu', padding='same')(m)
    #Bottleneck
    mNew3 = Convolution3D(16, (1, 1, 1), activation='relu', padding='same')(m)
    mNew3 = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(mNew3)
    mNew3 = Convolution3D(16, (1, 1, 1), padding='same')(mNew3)
    mNew3 = Dropout(0.7)(mNew3)
    #mMerge3 = merge([shortcut3, mNew3], mode='concat', concat_axis=-1)
    mMerge3 = Concatenate(axis=-1)([shortcut3, mNew3])
    
    m = Activation('relu')(mMerge3)

    shortcut4 = Convolution3D(16, (3, 3, 3), activation='relu', padding='same')(m)
     #Bottleneck
    mNew4 = Convolution3D(16, (1, 1, 1), activation='relu', padding='same')(m)
    mNew4 = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(mNew4)
    mNew4 = Convolution3D(16, (1, 1, 1), padding='same')(mNew4)
    mNew4 = Dropout(0.7)(mNew4)
    #mMerge4 = merge([shortcut4, mNew4], mode='concat', concat_axis=-1)
    mMerge4 = Concatenate(axis=-1)([shortcut4, mNew4])

    m = Activation('relu')(mMerge4)

    
    return m


def rpn(base_layers, num_anchors):
    x = Convolution3D(512, (3, 3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)
    x_class = Convolution3D(num_anchors, (1, 1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class')(x)
    x_regr = Convolution3D(num_anchors * 4, (1, 1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress')(x)
    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    
    if K.backend() == 'tensorflow':
        pooling_regions = 3
        input_shape = (num_rois, 3, 3, 3, 32) # The last one is nb_channels outputs from base_layers
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 32, 7, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

