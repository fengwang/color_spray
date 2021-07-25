__all__ = ['color_spray']

import os
import numpy as np
import imageio
import os.path
from os.path import expanduser

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.keras import backend as K

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

'''
    Example:
        model = read_model( './cached_folder' )
'''
def read_model(directory):
    #weights_path = f'{directory}/weights.h5'
    weights_path = os.path.join( directory, 'weights.h5' )
    if not os.path.isfile(weights_path):
        print( f'Failed to find weights from file {weights_path}' )
        return None

    #json_path = f'{directory}/js.json'
    json_path = os.path.join( directory, 'js.json' )
    if not os.path.isfile(json_path):
        print( f'Failed to find model from file {json_path}' )
        return None

    js_file = open( json_path, 'r' )
    model_json = js_file.read()
    js_file.close()
    model = model_from_json( model_json )
    model.load_weights( weights_path )
    return model

# credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({'InstanceNormalization': InstanceNormalization})


def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


def clean_input( image ):
    if len(image.shape) == 3:
        return image[:,:,0]

    if len(image.shape) == 2: # GRAY -> RGB
        return image.reshape( image.shape + (1,) )

    assert False, f"Unknown image with shape {image.shape}"

color_spray_model = None
def color_spray( input_gray_image_path, output_rgb_image_path=None ):
    # read gray image
    im = imageio.imread( input_gray_image_path )
    im = clean_input( im )

    # preparing input for the neural network
    row, col, _ = im.shape
    im = np.asarray( im, dtype='float32' )
    im = im / 127.5 - 1.0
    im = im.reshape( (1,)+im.shape )

    # prepare neural network
    global color_spray_model
    if color_spray_model is None:
        home = expanduser("~")
        model_dir = os.path.join( home, ".color_spray/model" )
        if not os.path.exists( model_dir ):
            assert False, f'Please download the model to {model_dir} from xxxxx'

        color_spray_model = read_model( model_dir )

    # predict high resolution image
    ans = color_spray_model.predict( im, batch_size=1 )
    ans = ans * 0.5 + 0.5
    ans = np.asarray( np.squeeze( ans ) * 255, dtype='uint8' )

    # save high resolution image
    if output_rgb_image_path is not None:
        imageio.imwrite( output_rgb_image_path, ans )

    # return high resolution image
    return ans

