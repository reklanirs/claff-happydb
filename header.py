'''
pip install tensorflow-hub pandas pysnooper sklearn pytablewriter matplotlib seaborn
'''

import os,re,sys,time,datetime,pathlib,platform


import math,random
from math import floor,ceil
import subprocess,shlex
import unicodedata,pickle,json
import scipy.io as sio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import keras
from tensorflow.keras import backend as K

# from tensorflow import keras
# from tensorflow.keras import backend as K
# import tensorflow.keras.layers as layers
# from tensorflow.keras.layers import Layer


from keras.layers import Input, Flatten, Lambda, Masking, Activation, LeakyReLU, PReLU, Softmax, BatchNormalization
from keras.layers import Conv2D, Dense, Dropout, Embedding, LSTM, Bidirectional, CuDNNGRU, Conv1D
from keras.constraints import NonNeg, MaxNorm, MinMaxNorm, max_norm
from keras.models import Model
from keras import regularizers
import keras.layers as layers
from keras.layers import Layer
from keras.optimizers import Adam
import pysnooper


import pandas as pd
import sklearn

#import nltk
from tensorflow.keras.preprocessing.text import text_to_word_sequence
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from functools import reduce
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample, shuffle
from sklearn.utils import class_weight
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
import logging
from pytablewriter import MarkdownTableWriter


from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, MaxPooling1D

from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHead
from keras_multi_head import MultiHeadAttention


from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras




def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

root_folder = "C:/Users/kanoto/Dropbox/Experiment/claff-offmychest" if platform.system() == 'Windows' else '/Users/reklanirs/Dropbox/Experiment/claff-offmychest/'
# root_folder = get_script_path()


labels = ['agency', 'social', 'concepts']


module_elmo_url = 'https://tfhub.dev/google/elmo/3'


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      # tf.keras.metrics.AUC(name='auc'),
]

def emoji_replacement(s):
    def dashrepl(matchobj):
        print(emoji.demojize(matchobj.group(0)))
        return emoji.demojize(matchobj.group(0))
    return re.sub(emoji.get_emoji_regexp(), dashrepl, s)



from keras.backend import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

# def gelu(x):
#     return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})
get_custom_objects().update({'gelu': Activation(gelu)})

get_custom_objects().update({'leaky_relu': Activation(LeakyReLU(alpha=0.05))})




def markdown_output(xheader, yheader, matrix, table_name=None):
    writer = MarkdownTableWriter()
    writer.table_name = table_name
    writer.headers = [''] + list(xheader)
    writer.value_matrix = [ [yheader[i]] + list(line) for i,line in enumerate(matrix)   ]
    writer.write_table()
    return writer

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




class CustomizedDenseLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        print('## CustomizedDenseLayer input_shape: {}'.format(input_shape))
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        print('## CustomizedDenseLayer input_shape: {}'.format(input_shape))
        return (input_shape[0], self.output_dim)


class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
