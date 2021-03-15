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
import pickle
# from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings
from tensorflow import keras
import pandas as pd
#import nltk
from tensorflow.keras.preprocessing.text import text_to_word_sequence
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from keras_emo_models import *
# from fz_models import *

from keras_monitoring import model_monitoring
from functools import reduce
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.dummy import DummyClassifier

from scipy.stats import wilcoxon
import multiprocessing

# from elmo_git_original_token_test import bilm_token_embedding

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




def text_process(sentences):
    # tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    tokens = [text_to_word_sequence(sentence) for sentence in sentences]
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
    cover_len = 50

    print('\nMax token length: {}'.format(max_len))
    print('Min token length: {}'.format(min_len))
    print('Mean token length: %.3f'%(mean_len))
    print('Median token length: %.3f'%(np.median(lengths)))
    print('Number of unique tokens: {}'.format(unique_token_num))
    print('Total number of tokens: {}'.format(total_token_num))
    print('Percentage of sentences that over the mean token length: %.3f%%'%( sum(i >= mean_len for i in lengths)*100.0 / len(lengths) ))
    print('Modified cover length is: {}'.format(cover_len))
    print('Percentage of sentences that under the cover_len: %.3f%%\n\n'%( sum(i <= cover_len for i in lengths)*100.0 / len(lengths) ))
    return max_len,mean_len,cover_len,unique_token_num,total_token_num

def tokenizer(sentences, unique_token_num):
    t = tf.keras.preprocessing.text.Tokenizer(num_words=unique_token_num + 1, lower=False, oov_token='<UKN>')
    tokens = [text_to_word_sequence(sentence) for sentence in sentences]
    t.fit_on_texts(tokens)
    return t

def sequential_and_padding(t, sentences, cover_len):
    tokens = [text_to_word_sequence(sentence) for sentence in sentences]
    sequences = t.texts_to_sequences(tokens)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=cover_len, padding='post')
    return sequences

def sequences_reformat(t, sequences):
    d = dict( [i,word] for word, i in t.word_index.items() )
    emb = []
    for indx,s in enumerate(sequences):
        tmp = [ d[i] if i!=0 else '.' for i in s] #? Is it safe using '.' as padding char
        emb.append(' '.join(tmp).strip())
    emb = np.array(emb)
    return emb


max_len,mean_len,cover_len,unique_token_num,total_token_num = 0,0,0,0,0
def token_prepare(sentences):
    global max_len,mean_len,cover_len,unique_token_num,total_token_num
    max_len,mean_len,cover_len,unique_token_num,total_token_num = text_process(sentences)
    t = tokenizer(np.array(sentences), unique_token_num)
    sequences = sequential_and_padding(t, sentences, cover_len)
    sequences = sequences_reformat(t, sequences)
    return sequences



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
    return ra,rs,p1,p2


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


def elmo_tfhub_implementation(tokens_input):
    with tf.Graph().as_default():
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        embeddings = elmo(
            inputs={
            "tokens": tokens_input,
            "sequence_len": [len(i) for i in tokens_input]
            },
            signature="tokens",
            as_dict=True
        )["elmo"]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            ret = sess.run(embeddings)
            # print(ret)
            # print(ret.shape)
            # print(type(ret))
    return ret


def elmo_text(t, sequences):
    d = {}
    for word, i in t.word_index.items():
        d[i] = word
    emb = []
    for indx,s in enumerate(sequences):
        if indx%10==0:
            print(indx, "%.2f%%"%(indx*100./len(sequences)) )
        tmp = []
        for i in s:
            tmp.append(d[i] if i!=0 else "")
        emb.append(tmp)
    return np.array(emb)

def dump_to_file(embedding, filename="embedding.pickle"):
    _filename = filename
    outfile = open(_filename,'wb')
    pickle.dump(embedding,outfile)
    outfile.close()
    print('Dump finished.')
    pass

def load_from_dump(filename="embedding.pickle"):
    _filename = filename
    infile = open(_filename,'rb')
    embedding = pickle.load(infile)
    return embedding

def main_bk():
    df,df1,df2 = read_data()
    moment, agency, social, concepts = extract_columns(df, ['moment', 'agency', 'social', 'concepts'])
    concepts_dim, concepts = categorical(concepts)
    print("Dimension of concepts is: %d"%concepts_dim)

    # labels = np.array(list(map(list, zip(agency, social, concepts))))
    labels = list(zip(agency, social, concepts))

    sequences_train = token_prepare(moment)

    dimension = 1024
    # hyperparameters
    hyper_params = {'learning_rate': 1e-2,
                    'epochs': 10,
                    'batch_size': 256,
                    'maxlen': cover_len,
                    'embedding_dim': dimension,
                    'concepts_dim': concepts_dim,
                    'optimizer': 'adam',
                    'loss': 'categorical_crossentropy',
                    'elmo_tag': 'elmo', # 'default' [None,1024] or 'elmo' [None, 50, 1024] or 'word_emb' [None, 50, 512]
                    'embedding_trainable': True, # No use if elmo_tag=word_emb
                    'elmo_signature': 'default',  # 'signature' if input is (1,) strings or 'tokens' if input is (50,) tokens
    }

    X_train, X_test, y_train, y_test = train_test_split(sequences_train, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


    print(len(X_train), len(X_test))
    print(X_train[0], y_train[0])

    

    # CNN + MTL + Embedding usage:
    model_type = 'CNN_MTL_Embedding'
    lastest_checkpoint = 'checkpoint/20181202_170003'
    modify_mtl = lambda y: [np.array([[i] for i,j,l in y]), np.array([[j] for i,j,l in y]), np.array([np.array(l) for i,j,l in y])]
    y_train, y_val, y_test = modify_mtl(y_train), modify_mtl(y_val), modify_mtl(y_test)
    hyper_params['loss'] = 'binary_crossentropy'
    hyper_params['optimizer'] = 'rmsprop'

    
    model = keras_cnn_mtl_elmo_v3(hyper_params)

    # compile the model
    model.compile(optimizer=hyper_params['optimizer'], loss=hyper_params['loss'], \
        loss_weights={'agency':.25, 'social':.25, 'concepts':10.0},
        metrics=[precision_m, recall_m, f1_m])

    d = datetime.datetime.today()
    timestamp = d.strftime('%Y%m%d_%H%M%S')


    model_compile = True
    history = ''
    if model_compile:
        # checkpoint_path, log_path = create_save_folder(model_type, dimension)
        history = model.fit(X_train, y_train, 
            epochs=hyper_params['epochs'], 
            validation_data=(X_val, y_val),
            batch_size=hyper_params['batch_size']
            )
    else:
        saved_model_path = os.path.join(root_folder, lastest_checkpoint)
        print("loading the saved model: {}".format(saved_model_path))
        model = keras.models.load_model(saved_model_path)


    test_mtl(model, X_test, y_test)

    from ploting import plot_metrics
    plot_metrics(history)

    # Generate csv file on test data
    # dftest = generate_test_output(model, hmid, sequences_test)
    # dftest['moment'] = pd.Series(tweets_test)
    # dftest.to_csv(os.path.join(systemrun_path,timestamp + '_' + model_type+'_'+ str(dimension) + '.csv'), index=False)


manager = multiprocessing.Manager()
ra = manager.dict()
rs = manager.dict()
historys = manager.dict()

def worker(model, hyper_params, data, ra, rs, historys):
    multiprocessing.freeze_support()
    X_train, y_train, X_test, y_test = data
    # compile the model
    model.compile(optimizer=hyper_params['optimizer'], loss=hyper_params['loss'], \
        # loss_weights={'agency':.25, 'social':.25, 'concepts':12.},
        metrics=[precision_m, recall_m, f1_m])

    model_compile = True
    if model_compile:
        # checkpoint_path, log_path = create_save_folder(model_type, dimension)
        history = model.fit(X_train, y_train, 
            epochs=hyper_params['epochs'], 
            validation_data=(X_test, y_test),
            batch_size=hyper_params['batch_size']
            )
    else:
        saved_model_path = os.path.join(root_folder, lastest_checkpoint)
        print("loading the saved model: {}".format(saved_model_path))
        model = keras.models.load_model(saved_model_path)


    a1,s1,pa,ps = test_mtl(model, X_test, y_test)
    ra.append(a1)
    rs.append(s1)
    historys.append(history)
    pass


def main():
    df,df1,df2 = read_data()
    moment, agency, social, concepts = extract_columns(df, ['moment', 'agency', 'social', 'concepts'])
    concepts_dim, concepts = categorical(concepts)
    print("Dimension of concepts is: %d"%concepts_dim)

    # labels = np.array(list(map(list, zip(agency, social, concepts))))
    labels = np.array(list(zip(agency, social, concepts)))

    sequences_train = token_prepare(moment)

    dimension = 1024
    # hyperparameters
    hyper_params = {'n_splits': 10,
                    'learning_rate': 1e-2,
                    'epochs': 10,
                    'batch_size': 256,
                    'maxlen': cover_len,
                    'embedding_dim': dimension,
                    'concepts_dim': concepts_dim,
                    'optimizer': 'adam',
                    'loss': 'categorical_crossentropy',
                    'elmo_tag': 'elmo', # 'default' [None,1024] or 'elmo' [None, 50, 1024] or 'word_emb' [None, 50, 512]
                    'embedding_trainable': True, # No use if elmo_tag=word_emb
                    'elmo_signature': 'default',  # 'signature' if input is (1,) strings or 'tokens' if input is (50,) tokens

    }

    # X_train, X_test, y_train, y_test = train_test_split(sequences_train, labels, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    

    # ra,rs = [],[]
    # history = ''
    jobs = []

    kf = KFold(n_splits=hyper_params['n_splits'], random_state=None, shuffle=False)
    for train_index, test_index in kf.split(sequences_train):
        time.sleep(10)
        X_train, X_test = sequences_train[train_index], sequences_train[test_index]
        y_train, y_test = labels[train_index], labels[test_index] 
        

        # CNN + MTL + Embedding usage:
        model_type = 'CNN_MTL_Embedding'
        # lastest_checkpoint = 'checkpoint/20181202_170003'
        modify_mtl = lambda y: [np.array([[i] for i,j,l in y]), np.array([[j] for i,j,l in y]), np.array([np.array(l) for i,j,l in y])]
        y_train, y_test = modify_mtl(y_train), modify_mtl(y_test)
        hyper_params['loss'] = 'binary_crossentropy'
        hyper_params['optimizer'] = 'rmsprop'

        
        model = keras_cnn_mtl_elmo_v3(hyper_params)

        
        p = multiprocessing.Process(target=worker, args=(model,hyper_params,(X_train, y_train, X_test, y_test), ra,rs,historys))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    print(ra.values(), rs.values())

        
        # with open('tmp.pickle', 'wb') as fout:
        #     pickle.dump([moment, agency, social, concepts, test_index, y_test, pa, ps], fout)
        # exit(0)

    ram = np.mean(np.array(ra), axis=0)
    rsm = np.mean(np.array(rs), axis=0)

    print('\n\n################################\nCross validation results:')
    for ret in (ram,rsm):
        print("f1_score: %.3f"%ret[0])
        print("auc_score: %.3f"%ret[1])
        print("accuracy_score: %.3f"%ret[2])
        print("precision_score: %.3f"%ret[3])
        print("recall_score: %.3f"%ret[4])

    print('\nf1 score list for agency: {}\nf1 score list for social: {}\nauc score list for agency: {}\nauc score list for social: {}\n'\
        .format(ra[:,0], rs[:,0], ra[:,1], rs[:,1]))


    from ploting import plot_metrics
    plot_metrics(history)

    # Generate csv file on test data
    # dftest = generate_test_output(model, hmid, sequences_test)
    # dftest['moment'] = pd.Series(tweets_test)
    # dftest.to_csv(os.path.join(systemrun_path,timestamp + '_' + model_type+'_'+ str(dimension) + '.csv'), index=False)




if __name__ == '__main__':
    
    main()
