from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution3D, MaxPooling3D, ZeroPadding3D, \
    AveragePooling2D, TimeDistributed, Dropout, GlobalAveragePooling3D
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization
import keras

def get_weight_path():
    if K.image_data_format() == 'channels_first':
        return 'squeezenet_weights_th_dim_ordering_th_kernels_notop.h5'
    else:
        return 'squeezenet_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height, dense):
    def get_output_length(input_length):
        input_length = input_length
        return input_length // 4
    return get_output_length(width) // 16 * 15, get_output_length(height)// 16 * 15, get_output_length(dense)  // 16 * 15

def nn_base(input_tensor=None, trainable=False):
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

    conv1 = Convolution3D(64, (3, 3, 3), activation='relu', strides=(2, 2, 2), padding='same', name='conv1')(img_input)
    maxpool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool1')(conv1)

    fire2_squeeze = Convolution3D(16, (1, 1, 1), activation='relu', padding='same', name='fire2_squeeze')(maxpool1)
    #merge2 = fire(fire2_squeeze, 64)

    fire2_expand1 = Convolution3D(64, (1, 1, 1), activation='relu', padding='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Convolution3D(64, (3, 3, 3), activation='relu', padding='same', name='fire2_expand2')(fire2_squeeze)
    #merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])
    merge2 = keras.layers.concatenate([fire2_expand1, fire2_expand2], axis=-1)

    fire3_squeeze = Convolution3D(16, (1, 1, 1), activation='relu', padding='same', name='fire3_squeeze')(merge2)
    #merge3 = fire(fire3_squeeze, 64)
    fire3_expand1 = Convolution3D(64, (1, 1, 1), activation='relu', padding='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = Convolution3D(64, (3, 3, 3), activation='relu', padding='same', name='fire3_expand2')(fire3_squeeze)
    #merge3 = Concatenate(axis=-1)([fire3_expand1, fire3_expand2])
    merge3 = keras.layers.concatenate([fire3_expand1, fire3_expand2], axis=-1)


    fire4_squeeze = Convolution3D(32, (1, 1, 1), activation='relu', padding='same', name='fire4_squeeze')(merge3)
    fire4_expand1 = Convolution3D(128, (1, 1, 1), activation='relu', padding='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='fire4_expand2')(fire4_squeeze)
    #merge4 = Concatenate(axis=-1)([fire4_expand1, fire4_expand2])
    merge4 = keras.layers.concatenate([fire4_expand1, fire4_expand2], axis=-1)
    #merge4 = fire(fire4_squeeze, 128)
    maxpool4 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool4')(merge4)

    fire5_squeeze = Convolution3D(32, (1, 1, 1), activation='relu', padding='same', name='fire5_squeeze')(maxpool4)
    fire5_expand1 = Convolution3D(128, (1, 1, 1), activation='relu', padding='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='fire5_expand2')(fire5_squeeze)
    #merge5 = Concatenate(axis=-1)([fire5_expand1, fire5_expand2])
    merge5 = keras.layers.concatenate([fire4_expand1, fire4_expand2], axis=-1)

    fire6_squeeze = Convolution3D(48, (1, 1, 1), activation='relu', padding='same', name='fire6_squeeze')(merge5)
    fire6_expand1 = Convolution3D(192, (1, 1, 1), activation='relu',padding='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = Convolution3D(192, (3, 3, 3), activation='relu', padding='same', name='fire6_expand2')(fire6_squeeze)
    #merge6 = Concatenate(axis=-1)([fire6_expand1, fire6_expand2])
    merge6 = keras.layers.concatenate([fire6_expand1, fire6_expand2], axis=-1)

    fire7_squeeze = Convolution3D(48, (1, 1, 1), activation='relu', padding='same', name='fire7_squeeze')(merge6)
    fire7_expand1 = Convolution3D(192, (1, 1, 1), activation='relu', padding='same', name='fire7_expand1')(fire7_squeeze)
    fire7_expand2 = Convolution3D(192, (3, 3, 3), activation='relu', padding='same', name='fire7_expand2')(fire7_squeeze)
    #merge7 = Concatenate(axis=-1)([fire7_expand1, fire7_expand2])
    merge7 = keras.layers.concatenate([fire7_expand1, fire7_expand2], axis=-1)

    fire8_squeeze = Convolution3D(64, (1, 1, 1), activation='relu', padding='same', name='fire8_squeeze')(merge7)
    fire8_expand1 = Convolution3D(256, (1, 1, 1), activation='relu', padding='same', name='fire8_expand1')(fire8_squeeze)
    fire8_expand2 = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='fire8_expand2')(fire8_squeeze)
    #merge8 = Concatenate(axis=-1)([fire8_expand1, fire8_expand2])
    merge8 = keras.layers.concatenate([fire8_expand1, fire8_expand2], axis=-1)

    maxpool8 = MaxPooling3D(
        pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool8')(merge8)
    
    fire8_dropout = Dropout(0.5, name='fire8_dropout')(merge8)

    conv9 = Convolution3D(13, (1, 1, 1), activation='relu', padding='valid', name='conv9')(fire8_dropout)
    # global_avgpool9 = GlobalAveragePooling3D()(conv9)
    # softmax = Activation("softmax", name='softmax')(global_avgpool9)
    return conv9

def rpn(base_layers, num_anchors):
    x = Convolution3D(512, (3, 3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)
    x_class = Convolution3D(num_anchors, (1, 1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class')(x)
    x_regr = Convolution3D(num_anchors * 4, (1, 1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress')(x)
    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=13, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    if K.backend() == 'tensorflow':
        pooling_regions = 3
        input_shape = (num_rois, 3, 3, 3, 256) # The last one is nb_channels outputs from base_layers
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 256, 7, 7, 7)

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