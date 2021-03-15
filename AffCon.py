#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 
import os,re,sys,time,datetime,pathlib
import math,random
import subprocess,shlex
import unicodedata,pickle
import scipy.io as sio
import numpy as np
import tensorflow as tf
# from tensorflow import keras
import keras
import pandas as pd
#import nltk
from tensorflow.keras.preprocessing.text import text_to_word_sequence
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras_emo_models import *
from keras_monitoring import model_monitoring
from functools import reduce
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.dummy import DummyClassifier

from header import *

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

root_folder = "/Users/reklanirs/Dropbox/Experiment/claff-happydb/"
train_labeled_path = os.path.join(root_folder, "data/TRAIN/labeled_10k.csv")
train_unlabeled_path = os.path.join(root_folder, "data/TRAIN/unlabeled_70k.csv")
test_unlabeled_path = os.path.join(root_folder, "data/TEST/unlabeled_17k.csv")

systemrun_path = os.path.join(root_folder, 'systemrun')

df = []

def read_data():
    df = pd.read_csv(train_labeled_path)
    df1 = pd.read_csv(train_unlabeled_path)
    df2 = pd.read_csv(test_unlabeled_path)
    # print(df)
    print('Total number of tweets: {}'.format(len(df)))
    print("\nagency distribution:\n{}".format(df['agency'].value_counts()))
    print("\nsocial distribution:\n{}".format(df['social'].value_counts()))
    print("\nUnion distribution:\n{}".format(df.groupby(["agency", "social"]).size()))
    return df,df1,df2



def my_test(l):
    for i in l:
        if len(i) == 1:
            print("Length one: ", i)
        t = text_to_word_sequence(i)
        if len(t) == 1 or len(t) == 140:
            print("\tError here:")
            print("\t\t", t)
            print("\t\t", i)
    pass

def text_process(tweets):
    # tokens = [nltk.word_tokenize(sentence) for sentence in tweets]
    tokens = [text_to_word_sequence(sentence) for sentence in tweets]
    lengths = np.array([ len(i) for i in tokens ])
    max_len = lengths.max()
    min_len = lengths.min()
    mean_len = lengths.mean()
    print(max_len, mean_len)

    s = set()
    for i in tokens:
        s.update(i)
    unique_token_num = len(s)
    total_token_num = sum(lengths)

    cover_len = math.ceil(mean_len + 1.8 * lengths.std())

    print('\nMax token length: {}'.format(max_len))
    print('\nMin token length: {}'.format(min_len))
    print('Mean token length: %.3f'%(mean_len))
    print('Median token length: %.3f'%(np.median(lengths)))
    print('Number of unique tokens: {}'.format(unique_token_num))
    print('Total number of tokens: {}'.format(total_token_num))
    print('Percentage of tweets that over the mean token length: %.3f%%'%( sum(i >= mean_len for i in lengths)*100.0 / len(lengths) ))
    print('Modified cover length is: {}'.format(cover_len))
    print('Percentage of tweets that under the cover_len: %.3f%%'%( sum(i <= cover_len for i in lengths)*100.0 / len(lengths) ))
    return max_len,mean_len,cover_len,unique_token_num,total_token_num



def extract_columns(df, l):
    df['agency'] = df['agency'].str.replace('yes','1').str.replace('no','0').astype(np.int32)
    df['social'] = df['social'].str.replace('yes','1').str.replace('no','0').astype(np.int32)
    print(df.iloc[0])
    print("\ndata types:\n{}".format(df.dtypes))
    tmp = {}
    for i in l:
        tmp[i] = np.array(df[i].tolist())
    # tweets = np.array(df['moment'].tolist())
    # agency = np.array(df['agency'].tolist())
    # social = np.array(df['social'].tolist())
    # return tweets,agency,social
    ret = [tmp[i] for i in l]
    return ret

    # tokenizer = keras.preprocessing.text.Tokenizer(num_words=numof_unique_tokens, oov_token='<UNK>', lower=False)
    # tokenizer.fit_on_texts(tweets)
    # sequences = tokenizer.texts_to_sequences(tweets)
    # print(np.array(sequences))
    # word_index = tokenizer.word_index

    # data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen)

def extract_columns_testdata(df):
    print("\ndata types:\n{}".format(df.dtypes))
    hmid = np.array(df['hmid'].tolist())
    tweets = np.array(df['moment'].tolist())
    return hmid,tweets

def categorical(concepts):
    s = set()
    for i in concepts:
        s.update(i.split('|'))
    d={i:j for j,i in enumerate(s)}
    ret = []
    l = len(s)
    for i in concepts:
        zeros = np.zeros((l))
        for j in i.split('|'):
            zeros[d[j]] = 1.0
        ret.append(zeros)
    return l,np.array(ret)


def tokenizer(tweets, unique_token_num):
    t = tf.keras.preprocessing.text.Tokenizer(num_words=unique_token_num + 1, lower=False, oov_token='<UKN>')
    tokens = [text_to_word_sequence(sentence) for sentence in tweets]
    t.fit_on_texts(tokens)
    # print(t.word_counts)
    # print(t.document_count)
    # for i,j in enumerate(t.word_index):
        # print(i,j.encode("utf-8"))
    # print(t.word_docs)
    return t

def sequential_and_padding(t, tweets, max_len):
    tokens = [text_to_word_sequence(sentence) for sentence in tweets]
    sequences = t.texts_to_sequences(tokens)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
    print(len(sequences))
    return sequences


def create_save_folder(model_type, dimension):
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    checkpoint_folder_path = os.path.abspath( os.path.join(os.path.curdir, 'checkpoint') )
    checkpoint_path = os.path.abspath( os.path.join(os.path.curdir, 'checkpoint', '%s_%s_%dD'%(timestamp, model_type, dimension) ) )
    log_path = os.path.abspath( os.path.join(os.path.curdir, 'log', timestamp) )
    if not os.path.exists(checkpoint_folder_path):
        pathlib.Path(checkpoint_folder_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(log_path):
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    print('Model checkpoint save path : {}\nLog path: {}'.format(checkpoint_path, log_path))
    return checkpoint_path,log_path

def f1_acc_pre_recall(y,p):
    y = [int(i) for i in y]
    yp = [int(round(i)) for i in p]
    ret = []
    ret.append(f1_score(y, yp, average="macro"))
    print("f1_score: %.3f"%ret[-1])
    ret.append(roc_auc_score(y, p))
    print("auc_score: %.3f"%ret[-1])
    ret.append(accuracy_score(y, yp))
    print("accuracy_score: %.3f"%ret[-1])
    ret.append(precision_score(y, yp, average="macro"))
    print("precision_score: %.3f"%ret[-1])
    ret.append(recall_score(y, yp, average="macro"))
    print("recall_score: %.3f"%ret[-1])
    return ret

def twin_data_f1_acc_pre_recall(y,predictions):
    y1,y2 = [i for i,j in y],[j for i,j in y]
    p1,p2 = [i for i,j in predictions],[j for i,j in predictions]
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
    print(predictions[0][:10])
    print(predictions[0].shape)
    print(predictions[:10])

    # y_p = [[round(i),round(j)] for i,j in predictions]
    # acc1 = sum( round(i[0])==j[0] for i,j in zip(predictions, y) )*100.0/len(y)
    # acc2 = sum( round(i[1])==j[1] for i,j in zip(predictions, y) )*100.0/len(y)
    # print("Accuracy %.3f%% %.3f%%"%(acc1, acc2))
    # twin_data_f1_acc_pre_recall(y, predictions)
    y1,y2 = y[0],y[1]
    p1,p2 = [round(i[0]) for i in predictions[0]],[round(i[0]) for i in predictions[1]]
    print('Label Agency:')
    ra = f1_acc_pre_recall(y1,p1)
    print('\nLabel Social:')
    rs = f1_acc_pre_recall(y2,p2)
    return ra,rs


def generate_test_output(model, hmid, X):
    predictions = model.predict(X)
    # mtl: [array( xxx,1 ),array( xxx,1 )]   non_mtl: array(17215, 2)
    tmp = []
    # print(predictions)
    # print(predictions[0][:10])
    binary_check = lambda x: 'yes' if round(x)==1 else 'no'
    if type(predictions) is np.ndarray:
        # non_mtl. General cnn model:
        for i,j in zip(hmid, predictions):
            tmp.append(np.array([i,binary_check(j[0]),binary_check(j[1])]))
    else:
        # mtl model
        for i,j,k in zip(hmid, predictions[0], predictions[1]):
            tmp.append(np.array([i,binary_check(j[0]),binary_check(k[0])]))
    df = pd.DataFrame(tmp, columns=['hmid','agency','social'])
    return df


def dummy(X_train, y_train, X_test, y_test):
    strategies = ['stratified', 'most_frequent', 'prior', 'uniform']
    for s in strategies:
        print('\nStrategy: %s'%s)
        dummy_classifier = DummyClassifier(strategy=s)
        dummy_classifier.fit( X_train,y_train )
        p = dummy_classifier.predict(X_test)
        twin_data_f1_acc_pre_recall(y_test, p)
    pass


def dummy2(tweets, agency, social):
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(tweets, agency, social, test_size=0.2, random_state=42)
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


def main_bk():
    # for calculating the process time
    tic = time.process_time()


    df,df1,df2 = read_data()
    moment, agency, social, concepts = extract_columns(df, ['moment', 'agency', 'social', 'concepts'])
    concepts_dim, concepts = categorical(concepts)
    print("Dimension of concepts is: %d"%concepts_dim)

    # labels = np.array(list(map(list, zip(agency, social, concepts))))
    labels = list(zip(agency, social, concepts))

    hmid,tweets_test = extract_columns_testdata(df2)

    my_test(np.concatenate((moment, tweets_test)))
    max_len,mean_len,cover_len,unique_token_num,total_token_num = text_process(np.concatenate((moment, tweets_test)))
    t = tokenizer(np.concatenate((moment, tweets_test)), unique_token_num)
    
    sequences_train = sequential_and_padding(t, moment, cover_len)
    sequences_test = sequential_and_padding(t, tweets_test, cover_len)

    # t, sequences = tokenizer_and_padding(tweets, unique_token_num, cover_len)

    dimension = 100
    # hyperparameters
    hyper_params = {'learning_rate': 1e-5,
                    'epochs': 10,
                    'batch_size': 1024,
                    'maxlen': cover_len,
                    'embedding_dim': dimension,
                    'concepts_dim': concepts_dim,
                    'max_words': unique_token_num + 2,
                    'optimizer': 'adam',
                    'loss': 'categorical_crossentropy',
                    'loss_weights': {'agency':.1, 'social':.1, 'concepts':0},
                    'train_able' : True
                    }

    embedding_matrix_path = os.path.join(root_folder, 'glove_embedding_matrix_{}.pickle'.format(dimension))
    embeddings_index = dict()
    if os.path.isfile(embedding_matrix_path):
        with open(embedding_matrix_path, 'rb') as fin:
            embedding_matrix = pickle.load(fin)
    else:
        f = open(root_folder + 'glove.twitter.27B/glove.twitter.27B.%dd.txt'%dimension, 'r', encoding="utf-8")
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
        with open(embedding_matrix_path, 'wb') as fout:
            pickle.dump(embedding_matrix, fout)

    hyper_params['weights'] = embedding_matrix


    X_train, X_test, y_train, y_test = train_test_split(sequences_train, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


    # # general CNN model
    # model_type = 'CNN'
    # lastest_checkpoint = 'checkpoint/'
    # model = keras_cnn(hyper_params)

    # # CNN + MTL usage:
    # model_type = 'CNN_MTL'
    # lastest_checkpoint = 'checkpoint/20181129_002449'
    # modify_mtl = lambda y: [np.array([[i] for i,j,l in y]), np.array([[j] for i,j,l in y]), np.array([np.array(l) for i,j,l in y])]
    # y_train, y_val, y_test = modify_mtl(y_train), modify_mtl(y_val), modify_mtl(y_test)
    # hyper_params['loss'] = 'binary_crossentropy'
    # hyper_params['optimizer'] = 'rmsprop'
    # model = keras_cnn_mtl_v2(hyper_params)

    # CNN + MTL + Embedding usage:
    model_type = 'CNN_MTL_Embedding'
    lastest_checkpoint = 'checkpoint/20181202_170003'
    modify_mtl = lambda y: [np.array([[i] for i,j,l in y]), np.array([[j] for i,j,l in y]), np.array([np.array(l) for i,j,l in y])]
    y_train, y_val, y_test = modify_mtl(y_train), modify_mtl(y_val), modify_mtl(y_test)
    hyper_params['loss'] = 'binary_crossentropy'
    hyper_params['optimizer'] = 'rmsprop'

    


    model = keras_cnn_mtl_emb_v3(hyper_params)

    # compile the model
    model.compile(optimizer=hyper_params['optimizer'], loss=hyper_params['loss'], \
        loss_weights=hyper_params['loss_weights'], \
        metrics=[precision_m, recall_m, f1_m])

    d = datetime.datetime.today()
    timestamp = d.strftime('%Y%m%d_%H%M%S')

    # print('Train data shape: X:{}, y:{}'.format(X_train.shape, y_train.shape))
    # print('Val data shape: X:{}, y:{}'.format(X_val.shape, y_val.shape))
    # print('Test data shape: X:{}, y:{}'.format(X_test.shape, y_test.shape))

    model_compile = True
    history = ''
    if model_compile:
        checkpoint_path, log_path = create_save_folder(model_type, dimension)
        history = model.fit(X_train, y_train, 
            epochs=hyper_params['epochs'], 
            batch_size=hyper_params['batch_size'],
            callbacks=model_monitoring(checkpoint_path, log_path),
            validation_data=(X_val, y_val))
    else:
        saved_model_path = os.path.join(root_folder, lastest_checkpoint)
        print("loading the saved model: {}".format(saved_model_path))
        model = keras.models.load_model(saved_model_path)
        # print("Test data shape: {} {}".format(X_test.shape, y_test.shape))

    # dummy(X_train, y_train, X_test, y_test)
    # dummy2(tweets, agency, social)

    if model_type == 'CNN':
        test(model, X_test, y_test)
    else:
        test_mtl(model, X_test, y_test)

    from ploting import plot_metrics
    plot_metrics(history)

    # Generate csv file on test data
    # dftest = generate_test_output(model, hmid, sequences_test)
    # dftest['moment'] = pd.Series(tweets_test)
    # dftest.to_csv(os.path.join(systemrun_path,timestamp + '_' + model_type+'_'+ str(dimension) + '.csv'), index=False)



def main():
    # for calculating the process time
    tic = time.process_time()


    df,df1,df2 = read_data()
    moment, agency, social, concepts = extract_columns(df, ['moment', 'agency', 'social', 'concepts'])
    concepts_dim, concepts = categorical(concepts)
    print("Dimension of concepts is: %d"%concepts_dim)

    # labels = np.array(list(map(list, zip(agency, social, concepts))))
    labels = np.array(list(zip(agency, social, concepts)))

    hmid,tweets_test = extract_columns_testdata(df2)

    my_test(np.concatenate((moment, tweets_test)))
    max_len,mean_len,cover_len,unique_token_num,total_token_num = text_process(np.concatenate((moment, tweets_test)))
    t = tokenizer(np.concatenate((moment, tweets_test)), unique_token_num)
    
    sequences_train = sequential_and_padding(t, moment, cover_len)
    sequences_test = sequential_and_padding(t, tweets_test, cover_len)

    # t, sequences = tokenizer_and_padding(tweets, unique_token_num, cover_len)


    dimension = 100
    # hyperparameters
    hyper_params = {'learning_rate': 1e-5,
                    'epochs': 10,
                    'batch_size': 1024,
                    'maxlen': cover_len,
                    'embedding_dim': dimension,
                    'concepts_dim': concepts_dim,
                    'max_words': unique_token_num + 2,
                    'optimizer': 'adam',
                    'loss': 'categorical_crossentropy',
                    'loss_weights': {'agency':.1, 'social':.1, 'concepts':0.},
                    'n_splits' : 5,
                    'train_able' : True
                    }


    embeddings_index = dict()
    f = open(root_folder + 'glove.twitter.27B/glove.twitter.27B.%dd.txt'%dimension, 'r', encoding="utf-8")
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


    # X_train, X_test, y_train, y_test = train_test_split(sequences_train, labels, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


    ra,rs = [],[]
    history = ''
    kf = KFold(n_splits=hyper_params['n_splits'], random_state=None, shuffle=True)
    for train_index, test_index in kf.split(sequences_train):
        X_train, X_test = sequences_train[train_index], sequences_train[test_index]
        y_train, y_test = labels[train_index], labels[test_index]


        # CNN + MTL + Embedding usage:
        model_type = 'CNN_MTL_Embedding'
        # lastest_checkpoint = 'checkpoint/20181202_170003'
        modify_mtl = lambda y: [np.array([[i] for i,j,l in y]), np.array([[j] for i,j,l in y]), np.array([np.array(l) for i,j,l in y])]
        y_train, y_test = modify_mtl(y_train), modify_mtl(y_test)
        hyper_params['loss'] = 'binary_crossentropy'
        hyper_params['optimizer'] = 'rmsprop'


        model = keras_cnn_mtl_emb_v3(hyper_params)

        # compile the model
        model.compile(optimizer=hyper_params['optimizer'], loss=hyper_params['loss'], \
            loss_weights=hyper_params['loss_weights'], \
            metrics=[precision_m, recall_m, f1_m])

        model_compile = True
        
        if model_compile:
            # checkpoint_path, log_path = create_save_folder(model_type, dimension)
            history = model.fit(X_train, y_train, 
                epochs=hyper_params['epochs'], 
                batch_size=hyper_params['batch_size'],
                # callbacks=model_monitoring(checkpoint_path, log_path),
                validation_data=(X_test, y_test))
        else:
            saved_model_path = os.path.join(root_folder, lastest_checkpoint)
            print("loading the saved model: {}".format(saved_model_path))
            model = keras.models.load_model(saved_model_path)

        a1,s1 = test_mtl(model, X_test, y_test)
        ra.append(a1)
        rs.append(s1)

    ra = np.mean(np.array(ra), axis=0)
    rs = np.mean(np.array(rs), axis=0)

    print('\n\n################################\nCross validation results:')
    for ret in (ra,rs):
        print("f1_score: %.3f"%ret[0])
        print("auc_score: %.3f"%ret[1])
        print("accuracy_score: %.3f"%ret[2])
        print("precision_score: %.3f"%ret[3])
        print("recall_score: %.3f"%ret[4])

    from ploting import plot_metrics
    plot_metrics(history)

    # Generate csv file on test data
    # dftest = generate_test_output(model, hmid, sequences_test)
    # dftest['moment'] = pd.Series(tweets_test)
    # dftest.to_csv(os.path.join(systemrun_path,timestamp + '_' + model_type+'_'+ str(dimension) + '.csv'), index=False)




if __name__ == '__main__':
    main()
