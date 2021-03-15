#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*- 
import os,re,sys,time,datetime,pathlib
import math,random
import subprocess,shlex
import unicodedata,pickle
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
#import nltk
from tensorflow.keras.preprocessing.text import text_to_word_sequence
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras_emo_models import *
from keras_monitoring import model_monitoring
from functools import reduce
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.dummy import DummyClassifier



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
clear = lambda: os.system('cls' if os.name=='nt' else 'clear')
#sys.stdin=open('in.txt','r')
''' concepts:
romance
education
food
shopping
entertainment
career
concepts
conversation
animals
religion
party
exercise
family
vacation
weather
technology
'''

root_folder = "/Users/reklanirs/Dropbox/NLP/CL-Aff Shared Task/claff-happydb-master/"
train_labeled_path = os.path.join(root_folder, "data/TRAIN/labeled_10k.csv")
train_unlabeled_path = os.path.join(root_folder, "data/TRAIN/unlabeled_70k.csv")
test_unlabeled_path = os.path.join(root_folder, "data/TEST/unlabeled_17k.csv")

df = []

def read_data():
    df = pd.read_csv(train_labeled_path)
    df1 = pd.read_csv(train_unlabeled_path)
    df2 = pd.read_csv(test_unlabeled_path)
    # print(df)
    print('Total number of tweets: {}'.format(len(df)))
    print("\nagency distribution:\n{}".format(df['agency'].value_counts()))
    print("\nsocial distribution:\n{}".format(df['social'].value_counts()))
    return df,df1,df2


def text_process(tweets):
    # tokens = [nltk.word_tokenize(sentence) for sentence in tweets]
    tokens = [text_to_word_sequence(sentence) for sentence in tweets]
    lengths = np.array([ len(i) for i in tokens ])
    max_len = lengths.max()
    mean_len = lengths.mean()
    print(max_len, mean_len)

    s = set()
    for i in tokens:
        s.update(i)
    unique_token_num = len(s)
    total_token_num = sum(lengths)

    cover_len = math.ceil(mean_len + 1.8 * lengths.std())

    print('\nMax token length: {}'.format(max_len))
    print('Mean token length: %.3f'%(mean_len))
    print('Number of unique tokens: {}'.format(unique_token_num))
    print('Total number of tokens: {}'.format(total_token_num))
    print('Percentage of tweets that over the mean token length: %.3f%%'%( sum(i >= mean_len for i in lengths)*100.0 / len(lengths) ))
    print('Modified cover length is: {}'.format(cover_len))
    print('Percentage of tweets that under the cover_len: %.3f%%'%( sum(i <= cover_len for i in lengths)*100.0 / len(lengths) ))
    return max_len,mean_len,cover_len,unique_token_num,total_token_num



def extract_columns(df):
    df['agency'] = df['agency'].str.replace('yes','1').str.replace('no','0').astype(np.int32)
    df['social'] = df['social'].str.replace('yes','1').str.replace('no','0').astype(np.int32)
    print(df.iloc[0])
    print("\ndata types:\n{}".format(df.dtypes))
    tweets = np.array(df['moment'].tolist())
    label1 = np.array(df['agency'].tolist())
    label2 = np.array(df['social'].tolist())
    return tweets,label1,label2

    # tokenizer = keras.preprocessing.text.Tokenizer(num_words=numof_unique_tokens, oov_token='<UNK>', lower=False)
    # tokenizer.fit_on_texts(tweets)
    # sequences = tokenizer.texts_to_sequences(tweets)
    # print(np.array(sequences))
    # word_index = tokenizer.word_index

    # data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen)

def tokenizer_and_padding(tweets, unique_token_num, max_len):
    t = tf.keras.preprocessing.text.Tokenizer(num_words=unique_token_num + 1, lower=False, oov_token='<UKN>')
    tokens = [text_to_word_sequence(sentence) for sentence in tweets]
    t.fit_on_texts(tokens)
    # print(t.word_counts)
    # print(t.document_count)
    # print(t.word_index)
    # print(t.word_docs)
    sequences = t.texts_to_sequences(tokens)
    print(sequences[:2])
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
    print(sequences[:2])
    print(len(sequences))
    return t,sequences


def create_save_folder():
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    checkpoint_folder_path = os.path.abspath( os.path.join(os.path.curdir, 'checkpoint') )
    checkpoint_path = os.path.abspath( os.path.join(os.path.curdir, 'checkpoint', timestamp) )
    log_path = os.path.abspath( os.path.join(os.path.curdir, 'log', timestamp) )
    if not os.path.exists(checkpoint_folder_path):
        pathlib.Path(checkpoint_folder_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(log_path):
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    print('Model checkpoint save path : {}\nLog path: {}'.format(checkpoint_path, log_path))
    return checkpoint_path,log_path

def f1_acc_pre_recall(y,p):
    print("f1_score: %.3f"%f1_score(y, p, average="macro"))
    print("accuracy_score: %.3f"%accuracy_score(y, p))
    print("precision_score: %.3f"%precision_score(y, p, average="macro"))
    print("recall_score: %.3f"%recall_score(y, p, average="macro"))

def twin_data_f1_acc_pre_recall(y,predictions):
    y1,y2 = [i for i,j in y],[j for i,j in y]
    p1,p2 = [round(i) for i,j in predictions],[round(j) for i,j in predictions]
    print('Label Agency:')
    f1_acc_pre_recall(y1,p1)
    print('\nLabel Social:')
    f1_acc_pre_recall(y2,p2)
    return predictions


def test(model, X, y):
    loss, acc = model.evaluate(X, y)
    print("loss and accuracy on test data: loss = {}, accuracy = {}".format(loss, acc))
    predictions = model.predict(X)
    # y_predict = np.argmax(predictions, axis=1)
    print('\n\n\n')
    print(type(predictions))
    print(predictions[:10])

    # y_p = [[round(i),round(j)] for i,j in predictions]
    # acc1 = sum( round(i[0])==j[0] for i,j in zip(predictions, y) )*100.0/len(y)
    # acc2 = sum( round(i[1])==j[1] for i,j in zip(predictions, y) )*100.0/len(y)
    # print("Accuracy %.3f%% %.3f%%"%(acc1, acc2))
    twin_data_f1_acc_pre_recall(y, predictions)
    pass

def test_mtl(model, X, y):
    # loss, acc = model.evaluate(X, y)
    # print("loss and accuracy on test data: loss = {}, accuracy = {}".format(loss, acc))
    predictions = model.predict(X)
    # y_predict = np.argmax(predictions, axis=1)
    print('\n\n\n')
    print(type(predictions))
    print(predictions[:10])

    # y_p = [[round(i),round(j)] for i,j in predictions]
    # acc1 = sum( round(i[0])==j[0] for i,j in zip(predictions, y) )*100.0/len(y)
    # acc2 = sum( round(i[1])==j[1] for i,j in zip(predictions, y) )*100.0/len(y)
    # print("Accuracy %.3f%% %.3f%%"%(acc1, acc2))
    # twin_data_f1_acc_pre_recall(y, predictions)

    y1,y2 = y[0],y[1]
    p1,p2 = [round(i[0]) for i in predictions[0]],[round(i[0]) for i in predictions[1]]
    print('p1')
    print(p1[:10])
    print('Label Agency:')
    f1_acc_pre_recall(y1,p1)
    print('\nLabel Social:')
    f1_acc_pre_recall(y2,p2)
    return predictions
    pass

def dummy(X_train, y_train, X_test, y_test):
    strategies = ['stratified', 'most_frequent', 'prior', 'uniform']
    for s in strategies:
        print('\nStrategy: %s'%s)
        dummy_classifier = DummyClassifier(strategy=s)
        dummy_classifier.fit( X_train,y_train )
        p = dummy_classifier.predict(X_test)
        twin_data_f1_acc_pre_recall(y_test, p)
    pass


def dummy2(tweets, label1, label2):
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(tweets, label1, label2, test_size=0.2, random_state=42)
    strategies = ['stratified', 'most_frequent', 'prior', 'uniform']
    for s in strategies:
        print('\nStrategy: %s'%s)
        dummy_classifier = DummyClassifier(strategy=s)
        dummy_classifier.fit( X_train,y1_train )
        p = dummy_classifier.predict(X_test)
        f1_acc_pre_recall(y1_test, p)

        dummy_classifier = DummyClassifier(strategy=s)
        dummy_classifier.fit( X_train,y2_train )
        p = dummy_classifier.predict(X_test)
        f1_acc_pre_recall(y2_test, p)
    pass

def main():
    df,df1,df2 = read_data()
    tweets, label1, label2 = extract_columns(df)
    labels = np.array(list(map(list, zip(label1, label2))))

    max_len,mean_len,cover_len,unique_token_num,total_token_num = text_process(tweets)
    t, sequences = tokenizer_and_padding(tweets, unique_token_num, cover_len)

    # X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(sequences, label1, label2, test_size=0.2, random_state=42)
    # print(len(X_train), len(X_test))
    # print(X_train[0], y1_train[0], y2_train[0])

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print(np.array(X_train).shape, np.array(X_test).shape, np.array(y_train).shape, np.array(y_test).shape )
    print(len(X_train), len(X_test))
    print(X_train[0], y_train[0])

    dimension = 50
    # hyperparameters
    hyper_params = {'learning_rate': 1e-1,
                    'epochs': 10,
                    'batch_size': 32,
                    'maxlen': cover_len,
                    'embedding_dim': dimension,
                    'max_words': unique_token_num + 2,
                    'optimizer': 'adam',
                    'loss': 'categorical_crossentropy'}
    # get embedding matrix

    # for calculating the process time
    tic = time.process_time()

    # # general CNN model
    # model = keras_cnn(hyper_params)

    # # CNN + MTL usage:
    # modify_mtl = lambda y: [np.array([[i] for i,j in y]), np.array([[j] for i,j in y])]
    # y_train, y_val, y_test = modify_mtl(y_train), modify_mtl(y_val), modify_mtl(y_test)
    # hyper_params['loss'] = 'binary_crossentropy'
    # hyper_params['optimizer'] = 'rmsprop'
    # model = keras_cnn_mtl(hyper_params)

    # CNN + MTL + Embedding usage:
    modify_mtl = lambda y: [np.array([[i] for i,j in y]), np.array([[j] for i,j in y])]
    y_train, y_val, y_test = modify_mtl(y_train), modify_mtl(y_val), modify_mtl(y_test)
    hyper_params['loss'] = 'binary_crossentropy'
    hyper_params['optimizer'] = 'rmsprop'

    embeddings_index = dict()
    
    f = open('/Users/reklanirs/Downloads/glove.twitter.27B/glove.twitter.27B.%dd.txt'%dimension, 'r', encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(t.word_index) + 1, dimension))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    hyper_params['weights'] = embedding_matrix
    hyper_params['train_able'] = True
    model = keras_cnn_mtl_emb(hyper_params)

    # compile the model
    model.compile(optimizer=hyper_params['optimizer'], loss=hyper_params['loss'], metrics=['accuracy'])

    d = datetime.datetime.today()
    timestamp = d.strftime('%Y%m%d_%H%M%S')

    # print('Train data shape: X:{}, y:{}'.format(X_train.shape, y_train.shape))
    # print('Val data shape: X:{}, y:{}'.format(X_val.shape, y_val.shape))
    # print('Test data shape: X:{}, y:{}'.format(X_test.shape, y_test.shape))

    model_compile = True
    if model_compile:
        checkpoint_path, log_path = create_save_folder()
        model.fit(X_train, y_train, 
            epochs=hyper_params['epochs'], 
            batch_size=hyper_params['batch_size'],
            callbacks=model_monitoring(checkpoint_path, log_path),
            validation_data=(X_val, y_val))
    else:
        saved_model_path = os.path.join(root_folder, 'checkpoint/20181123_015436')
        print("loading the saved model: {}".format(saved_model_path))
        model = keras.models.load_model(saved_model_path)
        # print("Test data shape: {} {}".format(X_test.shape, y_test.shape))
    # test(model, X_test, y_test)

    test_mtl(model, X_test, y_test)

    # dummy(X_train, y_train, X_test, y_test)
    # dummy2(tweets, label1, label2)



if __name__ == '__main__':
    main()
