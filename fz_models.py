#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 
import os,re,sys,time,datetime,pathlib
import math,random
import subprocess,shlex
import unicodedata,pickle
import scipy.io as sio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow import keras
# from tensorflow.keras import backend as K
# import tensorflow.keras.layers as layers
# from tensorflow.keras.layers import Layer
import keras
from keras.layers import Conv2D, Dense, Input, Flatten, Lambda, Masking, Activation
from keras.models import Model
from keras import backend as K
from keras import regularizers
import keras.layers as layers
from keras.layers import Layer

import pandas as pd
#import nltk
from tensorflow.keras.preprocessing.text import text_to_word_sequence
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from functools import reduce
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyClassifier
import logging

from tensor_trace_norm import TensorTraceNorm

from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPool2D, Concatenate, Reshape, Dropout


module_url = 'https://tfhub.dev/google/elmo/2'

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


class ElmoEmbeddingLayer(Layer):
    def __init__(self, tag, trainable, **kwargs):
        print('__init__ start')
        self.max_length = 50
        self.trainable=trainable
        self.tag=tag
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
        print('__init__ fin')

    def build(self, input_shape):
        print('build start')
        print(self.name)
        self.elmo = hub.Module(module_url, trainable=self.trainable, name="{}_module".format(self.name))
        print('hub load fin')
        if self.tag != 'word_emb':
            self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)
        print('build fin')

    def call(self, x, mask=None):
        print('call start')
        print(x.get_shape())
        result = self.elmo(tf.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )[self.tag]
        print('call fin')
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        if self.tag == 'default':
            return (input_shape[0], 1024)
        elif self.tag == 'elmo':
            return (input_shape[0], self.max_length, 1024)
        elif self.tag == 'word_emb':
            return (input_shape[0], self.max_length, 512)


class ElmoEmbeddingLayer_v2(Layer):
    def __init__(self, signature, tag, trainable, **kwargs):
        print('__init__ start')
        self.max_length = 50
        self.trainable = trainable
        self.tag = tag
        self.signature = signature
        super(ElmoEmbeddingLayer_v2, self).__init__(**kwargs)
        print('__init__ fin')

    def build(self, input_shape):
        print('build start')
        print(self.name)
        self.elmo = hub.Module(module_url, trainable=self.trainable, name="{}_module".format(self.name))
        print('hub load fin')
        print('Pre -- self.trainable_weights: {}'.format(self.trainable_weights))
        if self.tag != 'word_emb':
            self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name)) 
        print('Aft -- self.trainable_weights: {}'.format(self.trainable_weights))
        super(ElmoEmbeddingLayer_v2, self).build(input_shape)
        print('build fin')

    def call(self, x, mask=None):
        print('call start')
        print(x.get_shape())
        print(x.dtype)
        if self.signature == 'tokens':
            embeddings = self.elmo(
                inputs={
                    "tokens": x,
                    "sequence_len": [self.max_length for i in range(32)],
                },
                signature="tokens",
                as_dict=True)
        elif self.signature == 'default':
            embeddings = self.elmo(
                tf.squeeze(K.cast(x, tf.string), axis=1),
                signature='default',
                as_dict=True)
        print('call fin')
        return embeddings[self.tag]

    def compute_mask(self, inputs, input_mask=None):
        # return K.not_equal(inputs, '--PAD--')
        return None

    def compute_output_shape(self, input_shape):
        if self.tag == 'default':
            return (input_shape[0], 1024)
        elif self.tag == 'elmo':
            return (input_shape[0], self.max_length, 1024)
        elif self.tag == 'word_emb':
            return (input_shape[0], self.max_length, 512)




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


def keras_elmo(hyper_params):
    input_text = keras.Input(shape=(1,), dtype='string')
    print('input_text.shape',input_text.get_shape())
    embedding_layer = ElmoEmbeddingLayer(hyper_params['elmo_tag'], hyper_params['embedding_trainable'])(input_text)
    # Output_dim: [None, maxlen(50), 1024]
    # print('embedding_layer.shape', embedding_layer.get_shape())

    nonmasking = NonMasking()(embedding_layer)

    if hyper_params['elmo_tag'] == 'default': # [None, 1024]
        conv = keras.layers.Dense(128, activation=hyper_params['activation'])(nonmasking)
        # conv = Bidirectional(GRU(100, return_sequences=True))(nonmasking)
        pass
    elif hyper_params['elmo_tag'] == 'elmo' or hyper_params['elmo_tag'] == 'word_emb': # [None, 50, 1024]
        # conv = keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation=hyper_params['activation'], kernel_regularizer=regularizers.l2(0.01))(nonmasking) # input_shape=(50,1024),
        conv = Bidirectional(GRU(100, return_sequences=True))(nonmasking)
        #Output_dim: [None, 50, 64]

    # bn = keras.layers.BatchNormalization(axis=-1)(conv) #1 if hyper_params['elmo_tag'] == 'default' else 2
    drop = keras.layers.Dropout(hyper_params['dropout_rate'])(conv)
    print('drop.shape', drop.get_shape())
    #Output_dim: [None, 50, 64]

    if hyper_params['elmo_tag'] == 'default': # [None, 1024]
        conc = drop
    elif hyper_params['elmo_tag'] == 'elmo' or hyper_params['elmo_tag'] == 'word_emb': # [None, 50, 1024]
        # maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop)
        # flat = keras.layers.Flatten()(maxpool)
        avg_pool = GlobalAveragePooling1D()(drop)
        max_pool = GlobalMaxPooling1D()(drop)
        conc = concatenate([avg_pool, max_pool])
    #Output_dim: [None, 49, 64]

    dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    # dense = ''

    output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(i) for i in dense]
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model,dense


def keras_elmo_single(hyper_params):
    input_text = keras.Input(shape=(1,), dtype='string')
    print('input_text.shape',input_text.get_shape())
    embedding_layer = ElmoEmbeddingLayer(hyper_params['elmo_tag'], hyper_params['embedding_trainable'])(input_text)

    nonmasking = NonMasking()(embedding_layer)

    if hyper_params['elmo_tag'] == 'default': # [None, 1024]
        conv = keras.layers.Dense(128, activation=hyper_params['activation'])(nonmasking)
        # conv = Bidirectional(GRU(100, return_sequences=True))(nonmasking)
        pass
    elif hyper_params['elmo_tag'] == 'elmo' or hyper_params['elmo_tag'] == 'word_emb': # [None, 50, 1024]
        # conv = keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation=hyper_params['activation'], kernel_regularizer=regularizers.l2(0.01))(nonmasking) # input_shape=(50,1024),
        conv = Bidirectional(GRU(100, return_sequences=True))(nonmasking)
        #Output_dim: [None, 50, 64]

    bn = keras.layers.BatchNormalization(axis=-1)(conv)
    drop = keras.layers.Dropout(hyper_params['dropout_rate'])(bn)
    #Output_dim: [None, 50, 64]

    if hyper_params['elmo_tag'] == 'default': # [None, 1024]
        flat = drop
    elif hyper_params['elmo_tag'] == 'elmo' or hyper_params['elmo_tag'] == 'word_emb': # [None, 50, 1024]
        maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop)
        flat = keras.layers.Flatten()(maxpool)
    #Output_dim: [None, 49, 64]

    # dense = [CustomizedDenseLayer(output_dim=32)(flat) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly

    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(i) for i in dense]
    output = Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat)

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model



class ElmoEmbeddingLayer_v3(Layer):

    def __init__(self, output_dim=1024, trainable=True, **kwargs):
        self.output_dim = output_dim
        self.trainable = trainable
        self.elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        super(ElmoEmbeddingLayer_v3, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        # These are the 3 trainable weights for word_embedding, lstm_output1 and lstm_output2
        self.kernel1 = self.add_weight(name='kernel1',
                                       shape=(3,),
                                      initializer='uniform',
                                      trainable=self.trainable)
        # This is the bias weight
        self.kernel2 = self.add_weight(name='kernel2',
                                       shape=(),
                                      initializer='uniform',
                                      trainable=self.trainable)
        super(ElmoEmbeddingLayer_v3, self).build(input_shape)

    def call(self, x):
        # Get all the outputs of elmo_model
        model =  self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)
        
        # Embedding activation output  [None, 50,512]
        activation1 = model["word_emb"]
        
        # First LSTM layer outputs  [None, 50, 1024]
        activation2 = model["lstm_outputs1"]
        
        # Second LSTM layer output  [None, 50, 1024]
        activation3 = model["lstm_outputs2"]

        activation2 = tf.reduce_mean(activation2, axis=1)  # (None, 1024)
        activation3 = tf.reduce_mean(activation3, axis=1)  # (None, 1024)
        
        mul1 = tf.scalar_mul(self.kernel1[0], activation1)  # [None, 50,512]
        mul2 = tf.scalar_mul(self.kernel1[1], activation2)  # (None, 1024)
        mul3 = tf.scalar_mul(self.kernel1[2], activation3)  # (None, 1024)
        
        sum_vector = tf.add(mul2, mul3)   # (None, 1024)
        
        return tf.scalar_mul(self.kernel2, sum_vector)  # (50,)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def keras_cnn_mtl_emb_elmo_v3(hparam):
    if hparam['elmo_signature']=='tokens':
        input_text = keras.Input(shape=(hparam['maxlen'],), dtype=tf.string)
    elif hparam['elmo_signature']=='default':
        input_text = keras.Input(shape=(1,), dtype=tf.string)

    x = ElmoEmbeddingLayer_v3(1024, hparam['embedding_trainable'])(input_text)
    print('ElmoEmbeddingLayer_v3: {}'.format(x))

    
    # x = keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    # maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    # avgpool = keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
    # x = concatenate([avg_pool, max_pool])

    # flat_1 = keras.layers.Flatten()(maxpool_1)

    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    # x = Dense(256, activation='relu')(x)
    

    # indie1 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid')(x)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model



def keras_cnn_mtl_emb_glove(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    x = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'],
                                             weights=[hparam['weights']],
                                             trainable=hparam['train_able'])(input_text)

    x = SpatialDropout1D(0.3)(x)
    # conv = keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(embedding_layer)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])

    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid' if hparam['isDataBinary'] else 'relu')(conc)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model

def keras_cnn_mtl_emb_elmo(hparam):
    if hparam['elmo_signature']=='tokens':
        input_text = keras.Input(shape=(hparam['maxlen'],), dtype=tf.string)
    elif hparam['elmo_signature']=='default':
        input_text = keras.Input(shape=(1,), dtype=tf.string)

    x = ElmoEmbeddingLayer_v2(hparam['elmo_signature'], hparam['elmo_tag'], hparam['embedding_trainable'])(input_text)


    # maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    # avgpool = keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
    # x = keras.layers.Flatten()(x)

    if hparam['elmo_tag'] == 'word_emb':
        x = keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])

        # x = SpatialDropout1D(0.3)(x)
        # x = Bidirectional(GRU(100, return_sequences=True))(x)
        # avg_pool = GlobalAveragePooling1D()(x)
        # max_pool = GlobalMaxPooling1D()(x)
        # x = concatenate([avg_pool, max_pool])
    elif hparam['elmo_tag'] == 'default':
        x = Dense(32, activation='relu')(x)
    

    # indie1 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid' if hparam['isDataBinary'] else 'relu')(x)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model

def keras_cnn_v2_mtl_emb_elmo(hparam):
    if hparam['elmo_signature']=='tokens':
        input_text = keras.Input(shape=(hparam['maxlen'],), dtype=tf.string)
    elif hparam['elmo_signature']=='default':
        input_text = keras.Input(shape=(1,), dtype=tf.string)

    x = ElmoEmbeddingLayer_v2(hparam['elmo_signature'], hparam['elmo_tag'], hparam['embedding_trainable'])(input_text)


    # maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    # avgpool = keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
    # x = keras.layers.Flatten()(x)

    filter_size = [2,3,5]
    num_filters = 256
    if hparam['elmo_tag'] == 'word_emb':
        x = keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])

        # x = SpatialDropout1D(0.3)(x)
        # x = Bidirectional(GRU(100, return_sequences=True))(x)
        # avg_pool = GlobalAveragePooling1D()(x)
        # max_pool = GlobalMaxPooling1D()(x)
        # x = concatenate([avg_pool, max_pool])
        # x = Reshape((x.shape[1], x.shape[2], 1))(x)
        # x = tf.expand_dims(x, -1)

        # def expd(x):
        #     return K.expand_dims(x, -1)
        # x = Lambda(expd)(x)
        # conv = ['' for i in range(len(filter_size))]
        # maxpool = ['' for i in range(len(filter_size))]
        # for i in range(len(filter_size)):
        #     conv[i] = Conv2D(num_filters, kernel_size=(filter_size[i], 512), padding='valid', kernel_initializer='normal', activation='relu')(x)
        # for i in range(len(filter_size)):
        #     maxpool[i] = MaxPool2D(pool_size=(50 - filter_size[i] + 1, 1), strides=(1,1), padding='valid')(conv[i])
        # x = Concatenate(axis=-2)(maxpool)
        # x = Flatten()(x)
        # x = Dropout(0.3)(x)
    elif hparam['elmo_tag'] == 'default':
        x = Dense(256, activation='relu')(x)
    

    # indie1 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid' if hparam['isDataBinary'] else 'relu')(x)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model


def keras_basic(hparam):
    if hparam['elmo_signature']=='tokens':
        input_text = keras.Input(shape=(hparam['maxlen'],), dtype=tf.string)
    elif hparam['elmo_signature']=='default':
        input_text = keras.Input(shape=(1,), dtype=tf.string)

    x = ElmoEmbeddingLayer_v2(hparam['elmo_signature'], hparam['elmo_tag'], hparam['embedding_trainable'])(input_text)


    # maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    # avgpool = keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
    # x = keras.layers.Flatten()(x)

    if hparam['elmo_tag'] == 'word_emb':
        x = GlobalMaxPooling1D()(x)

        # x = SpatialDropout1D(0.3)(x)
        # x = Bidirectional(GRU(100, return_sequences=True))(x)
        # avg_pool = GlobalAveragePooling1D()(x)
        # max_pool = GlobalMaxPooling1D()(x)
        # x = concatenate([avg_pool, max_pool])
    elif hparam['elmo_tag'] == 'default':
        x = Dense(32, activation='relu')(x)
    

    # indie1 = keras.layers.Dense(16, activation='relu')(flat_1)

    x = Dense(1000, activation='relu')(x)
    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid' if hparam['isDataBinary'] else 'relu')(x)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model



def keras_basic_conv(hparam):
    if hparam['elmo_signature']=='tokens':
        input_text = keras.Input(shape=(hparam['maxlen'],), dtype=tf.string)
    elif hparam['elmo_signature']=='default':
        input_text = keras.Input(shape=(1,), dtype=tf.string)

    x = ElmoEmbeddingLayer_v2(hparam['elmo_signature'], hparam['elmo_tag'], hparam['embedding_trainable'])(input_text)


    # maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    # avgpool = keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
    # x = keras.layers.Flatten()(x)

    if hparam['elmo_tag'] == 'word_emb':
        x = keras.layers.Conv1D(filters=250, kernel_size=2, strides=1, padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x)

        # x = SpatialDropout1D(0.3)(x)
        # x = Bidirectional(GRU(100, return_sequences=True))(x)
        # avg_pool = GlobalAveragePooling1D()(x)
        # max_pool = GlobalMaxPooling1D()(x)
        # x = concatenate([avg_pool, max_pool])
    elif hparam['elmo_tag'] == 'default':
        x = Dense(32, activation='relu')(x)
    

    # indie1 = keras.layers.Dense(16, activation='relu')(flat_1)

    x = Dense(250, activation='relu')(x)
    x = Dropout(0.2)(x)
    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid' if hparam['isDataBinary'] else 'relu')(x)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model

