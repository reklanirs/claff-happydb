import tensorflow as tf
from tensorflow import keras


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

    output1 = keras.layers.Dense(1, activation='sigmoid')(indie1)
    output2 = keras.layers.Dense(1, activation='sigmoid')(indie1)

    model = keras.Model(inputs=input_text, outputs=[output1, output2])
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
