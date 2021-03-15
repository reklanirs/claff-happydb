import tensorflow as tf
import tensorflow_hub as hub
import sys



from traceback import print_stack,extract_stack

for x in extract_stack():
    if not x[0].startswith('<frozen importlib'):
        print('{} was imported by {}'.format(__name__, x[0]))
        if x[0] == 'AffCon.py':
            from tensorflow import keras
        elif x[0].startswith('AffCon_ELMo'):
            import keras
        break

# from tensorflow import keras
# from tensorflow.keras import backend as K
# import tensorflow.keras.layers as layers
# from tensorflow.keras.layers import Layer
import keras
from keras import backend as K
import keras.layers as layers
from keras.layers import Layer


# module_url = '/Users/reklanirs/workspace/NLP/Project/elmo_cache/9bb74bc86f9caffc8c47dd7b33ec4bb354d9602d'
module_url = 'https://tfhub.dev/google/elmo/3'


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



# class ElmoEmbeddingLayer(Layer):
#     def __init__(self, **kwargs):
#         print('__init__ start')
#         self.dimensions = 1024
#         self.trainable=True
#         super(ElmoEmbeddingLayer, self).__init__(**kwargs)
#         print('__init__ fin')

#     def build(self, input_shape):
#         print('build start')
#         print(self.name)
#         self.elmo = hub.Module(module_url, trainable=self.trainable, name="{}_module".format(self.name))
#         print('hub load fin')
#         print('self.trainable_weights type: {}'.format(self.trainable_weights))
#         self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoEmbeddingLayer, self).build(input_shape)
#         print('build fin')

#     def call(self, x, mask=None):
#         print('call start')
#         print(x.get_shape())
#         result = self.elmo(tf.squeeze(tf.cast(x, tf.string), axis=1),
#                       as_dict=True,
#                       signature='default',
#                       )['default']
#         print('call fin')
#         return result

#     def compute_mask(self, inputs, mask=None):
#         return K.not_equal(inputs, '--PAD--')

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.dimensions)


# class ElmoEmbeddingLayer_v3(Layer):
#     def __init__(self, **kwargs):
#         print('__init__ start')
#         self.dimensions = 1024
#         self.trainable=True
#         super(ElmoEmbeddingLayer_v3, self).__init__(**kwargs)
#         print('__init__ fin')

#     def build(self, input_shape):
#         print('build start')
#         print(self.name)
#         self.elmo = hub.Module(module_url, trainable=self.trainable, name="{}_module".format(self.name))
#         print('hub load fin')
#         # self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoEmbeddingLayer_v3, self).build(input_shape)
#         print('build fin')

#     def call(self, x, mask=None):
#         print('call start')
#         print(x.get_shape())
#         result = self.elmo(inputs={
#                 "tokens": x,
#                 "sequence_len": [29]
#             },
#             as_dict=True,
#             signature="tokens",
#         )['elmo']
#         print('call fin')
#         return result

#     def compute_mask(self, inputs, mask=None):
#         return K.not_equal(inputs, '--PAD--')

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.dimensions)



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
            self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))\
             if sys.platform.startswith('darwin') else K.tensorflow_backend.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
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





def keras_cnn_mtl_elmo(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],hparam['embedding_dim']))

    conv_1 = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(
        input_text)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)

    indie1 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie2 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie3 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)
    output3 = keras.layers.Dense(hparam['concepts_dim'], activation='sigmoid', name='concepts')(indie3)

    model = keras.Model(inputs=input_text, outputs=[output1, output2, output3])
    # print the model summary
    print(model.summary())
    return model

def keras_cnn_mtl_elmo_v2(hparam):
    input_text = keras.Input(shape=(1,), dtype='string')
    print('input_text.shape',input_text.get_shape())

    embedding_layer = ElmoEmbeddingLayer(hparam['elmo_tag'], hparam['embedding_trainable'])(input_text)
    embedding_layer = NonMasking()(embedding_layer)

    # print('embedding_layer.shape', embedding_layer.get_shape())
    dense = keras.layers.Dense(256, activation='relu')(embedding_layer)

    dense_node = 256
    indie1 = keras.layers.Dense(dense_node, activation='relu')(dense)
    indie2 = keras.layers.Dense(dense_node, activation='relu')(dense)
    indie3 = keras.layers.Dense(dense_node, activation='relu')(dense)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)
    output3 = keras.layers.Dense(hparam['concepts_dim'], activation='sigmoid', name='concepts')(indie3)

    model = keras.Model(inputs=input_text, outputs=[output1, output2, output3])
    # print the model summary
    print(model.summary())
    return model



def keras_cnn_mtl_elmo_v3(hparam):
    input_text = keras.Input(shape=(1,), dtype='string')
    print('input_text.shape',input_text.get_shape())

    embedding_layer = ElmoEmbeddingLayer(hparam['elmo_tag'], hparam['embedding_trainable'])(input_text)
    embedding_layer = NonMasking()(embedding_layer)

    # embedding_layer = keras.layers.Reshape([None,29,1024])(embedding_layer)
    print('embedding_layer.shape', embedding_layer.get_shape())
    
    conv_1 = keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu')(
        embedding_layer)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)

    dense_node = 256
    indie1 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie2 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie3 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)
    output3 = keras.layers.Dense(hparam['concepts_dim'], activation='sigmoid', name='concepts')(indie3)

    model = keras.Model(inputs=input_text, outputs=[output1, output2, output3])
    # print the model summary
    print(model.summary())
    return model


def keras_cnn_mtl_bert(hparam):
    # input_text = keras.Input(shape=(1,), dtype='string')
    input_text = keras.Input(shape=hparam['input_shape'], dtype='float')
    print('input_text.shape',input_text.get_shape())

    conv_1 = keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu')(
        input_text)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)

    dense_node = 256
    indie1 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie2 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie3 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)
    output3 = keras.layers.Dense(hparam['concepts_dim'], activation='sigmoid', name='concepts')(indie3)

    model = keras.Model(inputs=input_text, outputs=[output1, output2, output3])
    # print the model summary
    print(model.summary())
    return model









# Older models


def keras_cnn(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    embedding_layer = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'])(input_text)

    conv_1 = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(
        embedding_layer)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)
    output = keras.layers.Dense(2, activation='softmax', name='emotion')(flat_1)
    model = keras.Model(inputs=input_text, outputs=output)
    # print the model summary
    print(model.summary())
    return model



def keras_cnn_mtl(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    embedding_layer = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'])(input_text)

    conv_1 = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(
        embedding_layer)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)

    indie1 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie2 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)

    model = keras.Model(inputs=input_text, outputs=[output1, output2])
    # print the model summary
    print(model.summary())
    return model



def keras_cnn_mtl_emb(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    embedding_layer = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'],
                                             weights=[hparam['weights']],
                                             trainable=hparam['train_able'])(input_text)

    conv_1 = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(
        embedding_layer)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)

    indie1 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie2 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)

    model = keras.Model(inputs=input_text, outputs=[output1, output2])
    # print the model summary
    print(model.summary())
    return model

def keras_cnn_mtl_v2(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    embedding_layer = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'])(input_text)

    conv_1 = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(
        embedding_layer)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)

    indie1 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie2 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie3 = keras.layers.Dense(32, activation='relu')(flat_1)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)
    output3 = keras.layers.Dense(hparam['concepts_dim'], activation='sigmoid', name='concepts')(indie3)

    model = keras.Model(inputs=input_text, outputs=[output1, output2, output3])
    # print the model summary
    print(model.summary())
    return model

def keras_cnn_mtl_emb_v2(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    embedding_layer = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'],
                                             weights=[hparam['weights']],
                                             trainable=hparam['train_able'])(input_text)

    conv_1 = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(
        embedding_layer)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)

    indie1 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie2 = keras.layers.Dense(16, activation='relu')(flat_1)
    indie3 = keras.layers.Dense(16, activation='relu')(flat_1)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)
    output3 = keras.layers.Dense(hparam['concepts_dim'], activation='sigmoid', name='concepts')(indie3)

    model = keras.Model(inputs=input_text, outputs=[output1, output2, output3])
    # print the model summary
    print(model.summary())
    return model


def keras_cnn_mtl_emb_v3(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    embedding_layer = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'],
                                             weights=[hparam['weights']],
                                             trainable=hparam['train_able'])(input_text)

    conv_1 = keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu')(
        embedding_layer)
    drop_1 = keras.layers.Dropout(0.2)(conv_1)
    maxpool_1 = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(drop_1)
    flat_1 = keras.layers.Flatten()(maxpool_1)

    dense_node = 256
    indie1 = keras.layers.Dense(dense_node, activation='relu')(flat_1)
    indie2 = keras.layers.Dense(dense_node, activation='relu')(flat_1)
    indie3 = keras.layers.Dense(dense_node, activation='relu')(flat_1)

    output1 = keras.layers.Dense(1, activation='sigmoid', name='agency')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='social')(indie2)
    output3 = keras.layers.Dense(hparam['concepts_dim'], activation='sigmoid', name='concepts')(indie3)

    model = keras.Model(inputs=input_text, outputs=[output1, output2, output3])
    # print the model summary
    print(model.summary())
    return model

def keras_bilstm(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    embedding_layer = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'])(input_text)

    bilstm_1 = keras.layers.Bidirectional(keras.layers.LSTM(128))(embedding_layer)
    output = keras.layers.Dense(2, activation='softmax', name='emotion')(bilstm_1)
    model = keras.Model(inputs=input_text, outputs=output)
    # print the model summary
    print(model.summary())

    return model
