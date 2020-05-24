
from __future__ import print_function
from deepexplain.tensorflow import DeepExplain
import tensorflow as tf
import numpy as np
import keras
from sklearn import metrics
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras import backend as K
from keras.engine.topology import Layer
from deep_contribution_score import *
from data_gen import data_generator
import os
from rfe_sensitivity import *
from numpy.random import seed
from keras import optimizers
from tensorflow import set_random_seed

class Hadamard(Layer):

    def __init__(self, **kwargs):
        super(Hadamard, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True)
        super(Hadamard, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # print(x.shape, self.kernel.shape)
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        # print(input_shape)
        return input_shape

class DenseLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(DenseLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)