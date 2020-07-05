# -*- coding: utf-8 -*-
'''
3d-vgg
'''

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution3D, MaxPooling3D, ZeroPadding3D, \
    AveragePooling2D, TimeDistributed, Dropout

from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization


def get_weight_path():
    if K.image_data_format() == 'channels_first':
        return '3dnet_weights_th_dim_ordering_th_kernels_notop.h5'
    else:
        return '3dnet_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height, dense):
    def get_output_length(input_length):
        input_length = input_length // 4
        return input_length
    return get_output_length(width), get_output_length(height), get_output_length(dense)


def nn_base(input_tensor=None, trainable=False):
    # Determine proper input shape
    #input_shape=(image_size, image_size, image_size, num_channels=1) # l, h, w, c
    if K.image_data_format() == 'channels_first':
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

    if K.image_data_format() == 'channels_last':
        bn_axis = 4
    else:
        bn_axis = 1

    m = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(img_input)
    m = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(m)
    m = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(m)
    m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)

    m = Convolution3D(128, (3, 3, 3), activation='relu', padding='same')(m)
    m = Convolution3D(128, (3, 3, 3), activation='relu', padding='same')(m)
    #m = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(m)   

    '''
    m = Flatten(name='flatten')(m)
    m = Dense(1024, activation='relu', name='fc1')(m)
    m = Dropout(0.7)(m)
    
    m = Dense(1024, activation='relu', name='fc2')(m)
    m = Dropout(0.7)(m)   
    m = Dense(num_labels, activation='softmax')(m)
    '''
    
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
        input_shape = (num_rois, 3, 3, 3, 128) # The last one is nb_channels outputs from base_layers
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 128, 7, 7, 7)

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

