from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 5D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, depths, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, depths, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,z,r)
    # Output shape
        4D tensor with shape:
        `(1, num_rois, pool_size, pool_size, pool_size, channels)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][4] 

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x1 = rois[0, roi_idx, 0] # central point
            x2 = rois[0, roi_idx, 1]
            x3 = rois[0, roi_idx, 2]
            r = rois[0, roi_idx, 3]
            
            r_length = r / float(self.pool_size)

            num_pool_regions = self.pool_size

            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times

            #NOTE: Our 3D has long compile time, due to there is no 3d resize
            for kz in range(num_pool_regions):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1_min = x1 - r + ix * r_length * 2
                        x1_max = x1_min + r_length * 2
                        x2_min = x2 - r + jy * r_length * 2
                        x2_max = x2_min + r_length * 2
                        x3_min = x3 - r + kz * r_length * 2
                        x3_max = x3_min + r_length * 2

                        x1_min = K.cast(x1_min, 'int32')
                        x1_max = K.cast(x1_max, 'int32')
                        x2_min = K.cast(x2_min, 'int32')
                        x2_max = K.cast(x2_max, 'int32')
                        x3_min = K.cast(x3_min, 'int32')
                        x3_max = K.cast(x3_max, 'int32')

                        x1_min = K.maximum(1,x1_min)
                        x1_min = K.minimum(input_shape[1]-2,x1_min)
                        x1_max = x1_min + K.maximum(2,x1_max-x1_min)
                        x1_max = K.minimum(input_shape[1], x1_max)

                        x2_min = K.maximum(1,x2_min)
                        x2_min = K.minimum(input_shape[2]-2,x2_min)
                        x2_max = x2_min + K.maximum(2,x2_max-x2_min)
                        x2_max = K.minimum(input_shape[2], x2_max)

                        x3_min = K.maximum(1,x3_min)
                        x3_min = K.minimum(input_shape[3]-2,x3_min)
                        x3_max = x3_min + K.maximum(2,x3_max-x3_min)
                        x3_max = K.minimum(input_shape[3], x3_max)


                        new_shape = [input_shape[0], x1_max-x1_min, x2_max-x2_min, x3_max-x3_min, input_shape[4]]

                        x_crop = img[:, x1_min:x1_max, x2_min:x2_max, x3_min:x3_max,:]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2, 3))
                        outputs.append(pooled_val)
            

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 5, 2, 3, 4))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4, 5))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
