import os
import datetime
import numpy as np
import numpy_indexed as npi
import pandas as pd
from os.path import isfile, isdir
import matplotlib.pyplot as plt
import xlsxwriter as xls
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam, Adamax, Nadam
from keras.layers import Input, Bidirectional, LSTM, Embedding, RepeatVector, Dense, LeakyReLU, GRU, Dropout, Activation
from keras import Input, Model, optimizers
from keras_preprocessing.sequence import pad_sequences

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from yellowbrick.cluster import KElbowVisualizer


FILEPATH = 'D:\Cosas útiles\Ciber\TFM\Files\AutoVAS-master\dataset\token'
_SNIPPET_FILES = [
    FILEPATH + 'sard_result_0001_1000.txt',
    FILEPATH + 'sard_result_1001_2000.txt',
    FILEPATH + 'sard_result_2001_3000.txt',
    FILEPATH + 'sard_result_3001_4000.txt',
    FILEPATH + 'sard_result_4001_5000.txt',
    FILEPATH + 'sard_result_5001_6000.txt',
    FILEPATH + 'sard_result_6001_7000.txt',
    FILEPATH + 'sard_result_7001_8000.txt',
    FILEPATH + 'sard_result_8001_9283.txt',
    FILEPATH + 'nvd_result.txt'
]

_SOURCE_FILE    = 'source.txt'
_DATASET_FILE   = 'dataset.txt'
_CORPUS_FILE    = 'corpus.txt'
_W2V_MODEL_FILE = 'w2v.model'
_D2V_MODEL_FILE = 'd2v.model'
_S2V_MODEL_FILE = 's2v.model'
_FT_MODEL_FILE  = 'ft.model'
_GV_MODEL_FILE  = 'gv.model'

# for Embedding
CLASS_NUM = 2
MAX_WORDS = 200000
EMBEDDING_MODEL = 'w2v'
EMBEDDING_DIM = 300
SNIPPET_SIZE = 80

# for LSTM, GRU Model
TEST_SPLIT = 0.3
VALID_SPLIT = 0.2
K_FOLD = 5
BATCH_SIZE = 256
STATEFUL = False
HIDDEN_DIM = EMBEDDING_DIM
EPOCH_SIZE = 200 #200

# for Regularization
EARLYSTOP = False
L1_REG = 0.001
L2_REG = 0.001
DROPOUT_RATE = 0.42 #0.15 + np.random.rand() * 0.5

# for KMeans Model
OPT_K = True #To find the best number of clusters K
N_CLUSTERS = 7 #Else a static number is set
K_RANGE = range (2,10) #Range of clusters to be tested
DIFF_VALUE = 0.015 #Minimal improvement adding a cluster (1%)

# for MAIN method
SME = False
GROUP_FIRST = True
GET_CHARTS = False

AUTOENCOD = True
NORMAL_RUN = False

ONLY_VUL = False
MIN_NEUTRO = 0.85

embedding_matrix = np.zeros((0, 0))

def _create_data():
    print('[#] Create data file')
    with open(_DATASET_FILE, 'w') as wf:
        print('[-] Write dataset file:', _DATASET_FILE)
        wf.writelines("label\tsnippet\n")
        for snippet_file in tqdm(_SNIPPET_FILES):
            with open(snippet_file, 'r') as rf:
                lines = rf.readlines()
                for snippet in lines:
                    snippet = snippet[:-1].split('#')
                    t_len = len(snippet) - 1
                    label = snippet[0]
                    tokens = snippet[3:t_len]
                    wf.writelines(label + '\t' + ' '.join(tokens) + '\n')
    with open(_CORPUS_FILE, 'w') as wf:
        print('[-] Write corpus file:', _CORPUS_FILE)
        for snippet_file in tqdm(_SNIPPET_FILES):
            with open(snippet_file, 'r') as rf:
                lines = rf.readlines()
                for snippet in lines:
                    snippet = snippet[:-1].split('#')
                    t_len = len(snippet) - 1
                    tokens = snippet[3:t_len]
                    wf.write(' '.join(tokens) + '\n')

def _split_data(X, y, test_split, split_shuffle=True):
    print('[#] Split data (%.2f : %.2f)' % (1.0-test_split, test_split))
    if split_shuffle:
        ind = np.arange(len(X))
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]
    tr_size = int(len(X) * (1.0 - test_split))
    tr_X = X[:tr_size]
    te_X = X[tr_size:]
    tr_y = y[:tr_size]
    te_y = y[tr_size:]
    if STATEFUL:
        tr_size = int(int(int(tr_size * (1.0 - VALID_SPLIT)) / BATCH_SIZE) * BATCH_SIZE)
        tr_size += int(int(int(tr_size * VALID_SPLIT) / BATCH_SIZE) * BATCH_SIZE)
        # tr_size = int(int(tr_size / BATCH_SIZE) * BATCH_SIZE)
        te_size = int(int(len(te_X) / BATCH_SIZE) * BATCH_SIZE)
        tr_X = tr_X[:tr_size]
        te_X = te_X[:te_size]
        tr_y = tr_y[:tr_size]
        te_y = te_y[:te_size]
    print('[-] train data shape (X, y):', tr_X.shape, tr_y.shape)
    print('[-] test data shape (X, y):', te_X.shape, te_y.shape)
    return tr_X, tr_y, te_X, te_y


def _split_data_for_kfold(X, y, k_fold, split_shuffle=True):
    kf = KFold(n_splits=k_fold, shuffle=split_shuffle)
    if STATEFUL:
        kf_size = int(int(int(len(X)/BATCH_SIZE)/k_fold) * BATCH_SIZE) * k_fold
        print(kf_size)
        X = X[:kf_size]
        y = y[:kf_size]
    print('[-] data shape (X, y):', X.shape, y.shape)
    return X, y, kf

def _create_ft_model():
    print('[*] Create FastTest model')
    if not isfile(_CORPUS_FILE):
        print('[!] Please check the corpus file: %s' % _CORPUS_FILE)
        return
    corpus = list()
    with open(_CORPUS_FILE) as f:
        lines = f.readlines()
        for line in lines:
            temp = line[:-1].split(' ')
            corpus.append(temp[:len(temp)])
    print('[-] fasttext embedding ...')
    model = FastText(sentences=corpus, window=5, min_count=5, workers=4,
                     size=EMBEDDING_DIM, iter=200, sample=1e-4, sg=1, negative=5)
    print('[-] save fasttext model')
    model.save(_FT_MODEL_FILE)


def _create_w2v_model():
    print('[#] Create word2vec model')
    if not isfile(_CORPUS_FILE):
        print('[!] Please check the corpus file: %s' % _CORPUS_FILE)
        return
    corpus = list()
    with open(_CORPUS_FILE) as f:
        lines = f.readlines()
        for line in lines:
            temp = line[:-1].split(' ')
            corpus.append(temp[:len(temp)])
    print('[-] word2vec embedding ...')
    model = Word2Vec(sentences=corpus, window=5, min_count=5, workers=4,
                     size=EMBEDDING_DIM, iter=200, sample=1e-4, sg=1, negative=5)
    print('[-] save word2vec model')
    model.wv.save_word2vec_format(_W2V_MODEL_FILE)


def _create_d2v_model():
    print('[#] Create doc2vec model')
    if not isfile(_CORPUS_FILE):
        print('[!] Please check the corpus file: %s' % _CORPUS_FILE)
        return
    corpus = list()
    with open(_CORPUS_FILE) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            temp = line.split('\n')[0].strip()
            corpus.append(TaggedDocument(temp.split(' '), [idx]))
    print('[-] doc2vec embedding ...')
    model = Doc2Vec(documents=corpus, window=5, min_count=10, workers=4,
                    vector_size=EMBEDDING_DIM, epochs=200, sample=1e-4, negative=5, dm=0)
    print('[-] save doc2vec model')
    model.save(_D2V_MODEL_FILE)

def _create_s2v_model():
    print('[#] Create sent2vec model')
    if not isfile(_CORPUS_FILE):
        print('[!] Please check the corpus file: %s' % _CORPUS_FILE)
        return
    corpus = list()
    with open(_CORPUS_FILE) as f:
        lines = f.readlines()
        idx = 0
        for line in lines:
            temp = line.split('\n')[0].strip()
            sents = temp.split(';')
            for sent in sents:
                corpus.append(TaggedDocument(sent.split(' '), [idx]))
                idx += 1
    print('[-] sent2vec embedding ...')
    model = Doc2Vec(documents=corpus, window=5, min_count=10, workers=4,
                    vector_size=EMBEDDING_DIM, epochs=200, sample=1e-4, negative=5, dm=0)
    print('[-] save sent2vec model')
    model.save(_S2V_MODEL_FILE)


def _create_embedding_matrix(word_index, embed_opt='w2v'):
    print('[#] Create embedding matrix')
    words_size = min(MAX_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((words_size, EMBEDDING_DIM))
    if embed_opt == 'w2v':
        if not isfile(_W2V_MODEL_FILE): _create_w2v_model()
        word2vec = KeyedVectors.load_word2vec_format(_W2V_MODEL_FILE)
        cnt = 0
        for word, i in word_index.items():
            if word in word2vec.key_to_index:
                embedding_matrix[i] = word2vec.get_vector(word)
                cnt += 1
    elif embed_opt == 'gv':
        if not isfile(_GV_MODEL_FILE): return
        word2vec = KeyedVectors.load_word2vec_format(_GV_MODEL_FILE)
        cnt = 0
        for word, i in word_index.items():
            if word in word2vec.key_to_index:
                embedding_matrix[i] = word2vec.word_vec(word)
                cnt += 1
        print(cnt)
    elif embed_opt == 'ft':
        if not isfile(_FT_MODEL_FILE): _create_ft_model()
        ft = FastText.load(_FT_MODEL_FILE)
        cnt = 0
        for word, i in word_index.items():
            if word in ft.wv.key_to_index:
                embedding_matrix[i] = ft.wv[word]
                cnt += 1
        print(cnt)
    elif embed_opt == 'd2v':
        if not isfile(_D2V_MODEL_FILE): _create_d2v_model()
        doc2vec = Doc2Vec.load(_D2V_MODEL_FILE)
        for word, i in word_index.items():
            if word in doc2vec.wv.key_to_index:
                embedding_matrix[i] = doc2vec.wv.word_vec(word)
    elif embed_opt == 's2v':
        if not isfile(_S2V_MODEL_FILE): _create_s2v_model()
        sent2vec = Doc2Vec.load(_S2V_MODEL_FILE)
        for word, i in word_index.items():
            if word in sent2vec.wv.key_to_index:
                embedding_matrix[i] = sent2vec.wv.word_vec(word)
    return embedding_matrix


def load_snippet_data(embed_opt='w2v'):
    print('[*] Load snippet data')
    if not isfile(_DATASET_FILE): _create_data()
    df = pd.read_csv(_DATASET_FILE, sep='\t', usecols=['label', 'snippet'],
                     dtype={'label': int, 'snippet': str})
    df.fillna('', inplace=True)
    print('[-] tokenizing ...')
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df['snippet'])
    snippet = tokenizer.texts_to_sequences(df['snippet'])
    print('[-] post zero padding ... (size: %d)' % SNIPPET_SIZE)
    X = pad_sequences(snippet, maxlen=SNIPPET_SIZE, padding='post')
    df.loc[df['label'] == 2, 'label'] = 1
    y = to_categorical(df['label'], CLASS_NUM)
    print('[-] Start to oversampling using SMOTEENN')
    sme = SMOTEENN(random_state=42)
    X, y = sme.fit_resample(X, y)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    tr_X, tr_y, te_X, te_y = _split_data(X, y, TEST_SPLIT, split_shuffle=True)
    embedding_matrix = _create_embedding_matrix(tokenizer.word_index, embed_opt)
    print('[*] Done!!! -> load snippet data\n')
    return (tr_X, tr_y), (te_X, te_y), embedding_matrix

def alt_load_snippet_data(embed_opt='w2v'):
    print('[*] Load snippet data')
    if not isfile(_DATASET_FILE): _create_data()
    df = pd.read_csv(_DATASET_FILE, sep='\t', usecols=['label', 'snippet'],
                     dtype={'label': int, 'snippet': str})
    df.fillna('', inplace=True)
    print('[-] tokenizing ...')
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df['snippet'])
    snippet = tokenizer.texts_to_sequences(df['snippet'])
    print('[-] post zero padding ... (size: %d)' % SNIPPET_SIZE)
    X = pad_sequences(snippet, maxlen=SNIPPET_SIZE, padding='post')
    df.loc[df['label'] == 2, 'label'] = 1
    y = to_categorical(df['label'], CLASS_NUM)
    print('[-] Start to oversampling using SMOTEENN')
    sme = SMOTEENN(random_state=42)
    X, y = sme.fit_resample(X, y)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    #tr_X, tr_y, te_X, te_y = _split_data(X, y, TEST_SPLIT, split_shuffle=True)
    embedding_matrix = _create_embedding_matrix(tokenizer.word_index, embed_opt)
    print('[*] Done!!! -> load snippet data\n')
    return X, y, embedding_matrix

def wo_sme_load_snippet_data(embed_opt='w2v'):
    print('[*] Load snippet data')
    if not isfile(_DATASET_FILE): _create_data()
    df = pd.read_csv(_DATASET_FILE, sep='\t', usecols=['type', 'label', 'snippet'],
                     dtype={'type':str, 'label': int, 'snippet': str})
    df.fillna('', inplace=True)
    print('[-] tokenizing ...')
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df['snippet'])
    snippet = tokenizer.texts_to_sequences(df['snippet'])
    print('[-] post zero padding ... (size: %d)' % SNIPPET_SIZE)
    X = pad_sequences(snippet, maxlen=SNIPPET_SIZE, padding='post')
    df.loc[df['label'] == 2, 'label'] = 1
    y = to_categorical(df['label'], CLASS_NUM)
    t = df['type'].to_numpy()
    embedding_matrix = _create_embedding_matrix(tokenizer.word_index, embed_opt)
    print('[*] Done!!! -> load snippet data\n')
    return X, t, y, embedding_matrix

def t_load_snippet_data(embed_opt='w2v'):
    print ('[*] Load source data')
    if not isfile(_DATASET_FILE): _create_data()
    src = pd.read_csv(_DATASET_FILE, sep='\t', usecols=['label', 'snippet'],
                     dtype={'label': int, 'snippet': str})
    src.fillna('', inplace=True)
    print('[*] Load snippet data')
    if not isfile(_DATASET_FILE): _create_data()
    df = pd.read_csv(_DATASET_FILE, sep='\t', usecols=['label', 'snippet'],
                     dtype={'label': int, 'snippet': str})
    df.fillna('', inplace=True)
    print('[-] tokenizing ...')
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df['snippet'])
    snippet = tokenizer.texts_to_sequences(df['snippet'])
    print('[-] post zero padding ... (size: %d)' % SNIPPET_SIZE)
    X = pad_sequences(snippet, maxlen=SNIPPET_SIZE, padding='post')
    df.loc[df['label'] == 2, 'label'] = 1
    y = to_categorical(df['label'], CLASS_NUM)
    print('[-] Start to oversampling using SMOTEENN')
    sme = SMOTEENN(random_state=42)
    X, y = sme.fit_resample(X, y)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    #tr_X, tr_y, te_X, te_y = _split_data(X, y, TEST_SPLIT, split_shuffle=True)
    embedding_matrix = _create_embedding_matrix(tokenizer.word_index, embed_opt)
    print('[*] Done!!! -> load snippet data\n')
    return src, X, y, embedding_matrix

def load_snippet_data_for_kfold(embed_opt='w2v'):
    print('[*] Load snippet data for K-fold cross validation')
    if not isfile(_DATASET_FILE): _create_data()
    df = pd.read_csv(_DATASET_FILE, sep='\t', usecols=['label', 'snippet'],
                     dtype={'label': int, 'snippet': str})
    df.fillna('', inplace=True)
    print('[-] tokenizing ...')
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df['snippet'])
    snippet = tokenizer.texts_to_sequences(df['snippet'])
    print('[-] post zero padding ... (size: %d)' % SNIPPET_SIZE)
    X = pad_sequences(snippet, maxlen=SNIPPET_SIZE, padding='post')
    y = to_categorical(df['label'], CLASS_NUM)
    print('[-] Start to oversampling using SMOTEENN')
    sme = SMOTEENN(random_state=42)
    X, y = sme.fit_resample(X, y)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    X, y, kf = _split_data_for_kfold(X, y, k_fold=K_FOLD, split_shuffle=True)
    embedding_matrix = _create_embedding_matrix(tokenizer.word_index, embed_opt)
    print('[*] Done!!! -> load snippet data for K-fold cross validation\n')
    return X, y, kf, embedding_matrix


def add_embedding_layer(embedding_matrix):
    print('[#] add embedding layer...')
    model = Sequential()
    input_dim, output_dim = embedding_matrix.shape
    if STATEFUL:
        model.add(Embedding(input_dim, output_dim, weights=[embedding_matrix],
                            batch_input_shape=(BATCH_SIZE, SNIPPET_SIZE),
                            input_length=SNIPPET_SIZE, trainable=False))
    else:
        model.add(Embedding(input_dim, output_dim, weights=[embedding_matrix],
                            input_shape=(SNIPPET_SIZE,),
                            input_length=SNIPPET_SIZE, trainable=False))
    return model


def lstm(embedding_matrix):
    print('[*] Start to build lstm model')
    model = add_embedding_layer(embedding_matrix)
    model.add(LSTM(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                   kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    adam = Adamax(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    print('[*] Done!!! -> build lstm model\n')
    return model


def blstm(embedding_matrix):
    print('[*] Start to build bidirectional lstm model')
    model = add_embedding_layer(embedding_matrix)
    model.add(Bidirectional(LSTM(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                                 kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG))))
    
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    adam = Adamax(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, 
            metrics = [
               'accuracy',
               tf.keras.metrics.TruePositives(name='TP'),
               tf.keras.metrics.TrueNegatives(name='TN'),
               tf.keras.metrics.FalsePositives(name='FP'),
               tf.keras.metrics.FalseNegatives(name='FN'),
               tf.keras.metrics.AUC(name='auc')
           ])
    model.summary()
    print('[*] Done!!! -> build bidirectional lstm model\n')
    return model


def kmeans(n):
    if n==0:
        model = KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=300, n_init=100, random_state=None, tol=0.0001)
    else:
        model = KMeans(n_clusters=n, algorithm='lloyd', copy_x=True, init='k-means++', max_iter=300, n_init=100, random_state=None, tol=0.0001)
    print('[*] Kmeans model built with', n,'clusters\n')
    return model

def multi_lstm(embedding_matrix):
    print('[*] Start to build multi-layer lstm model')
    model = add_embedding_layer(embedding_matrix)
    model.add(LSTM(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                   kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                   return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                   kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                   return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                   kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                   return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    adam = Adamax(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    print('[*] Done!!! -> build multi-layer lstm model\n')
    return model

def multi_blstm(embedding_matrix):
    print('[*] Start to build multi-layer bidirectional lstm model')
    model = add_embedding_layer(embedding_matrix)
    model.add(Bidirectional(LSTM(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                                 kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                                 return_sequences=True)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                                 kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                                 return_sequences=True)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                                 kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                                 return_sequences=False)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    adam = Adamax(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    print('[*] Done!!! -> build multi-layer bidirectional lstm model\n')
    return model

def gru(embedding_matrix):
    print('[*] Start to build lstm model')
    model = add_embedding_layer(embedding_matrix)
    model.add(GRU(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                  kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    adam = Adamax(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    print('[*] Done!!! -> build lstm model\n')
    return model


def bgru(embedding_matrix):
    print('[*] Start to build bidirectional lstm model')
    model = add_embedding_layer(embedding_matrix)
    model.add(Bidirectional(GRU(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                                kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG))))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    adam = Adamax(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, 
            metrics = [
               'accuracy',
               tf.keras.metrics.TruePositives(name='TP'),
               tf.keras.metrics.TrueNegatives(name='TN'),
               tf.keras.metrics.FalsePositives(name='FP'),
               tf.keras.metrics.FalseNegatives(name='FN'),
               tf.keras.metrics.AUC(name='auc')
           ])
    model.summary()
    print('[*] Done!!! -> build bidirectional lstm model\n')
    return model


def multi_gru(embedding_matrix):
    print('[*] Start to build multi-layer lstm model')
    model = add_embedding_layer(embedding_matrix)
    model.add(GRU(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                  kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                  return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(GRU(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                  kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                  return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(GRU(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                  kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                  return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    adam = Adamax(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    print('[*] Done!!! -> build multi-layer lstm model\n')
    return model

def multi_bgru(embedding_matrix):
    print('[*] Start to build multi-layer bidirectional lstm model')
    model = add_embedding_layer(embedding_matrix)
    model.add(Bidirectional(GRU(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                                kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                                return_sequences=True)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(GRU(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                                kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                                return_sequences=True)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(GRU(HIDDEN_DIM, kernel_initializer='he_normal', stateful=STATEFUL,
                                kernel_regularizer=regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                                return_sequences=False)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    adam = Adamax(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    print('[*] Done!!! -> build multi-layer bidirectional lstm model\n')
    return model


def fit_and_result(model, tr_X, tr_y, te_X, te_y):
    print('[*] Start to fit model')
    if EARLYSTOP:
        early_stopping = EarlyStopping(monitor='loss', patience=10)
    else:
        early_stopping = None

    hist = model.fit(tr_X, tr_y, validation_split=VALID_SPLIT, epochs=EPOCH_SIZE,
                     batch_size=BATCH_SIZE, shuffle=False, callbacks=early_stopping)
    print('[-] make result report')
    pred_y = np.argmax(model.predict(te_X, batch_size=BATCH_SIZE), axis=1)
    te_y = np.argmax(te_y, axis=1)
    print('[*] Done!!! -> fit and result\n')
    rep = classification_report(te_y, pred_y, output_dict=True)
    rep.update({"accuracy": {"precision": None, "recall": None, "f1-score": rep["accuracy"], "support": rep['macro avg']['support']}})
    rep = pd.DataFrame(rep).transpose()
    return rep, hist

def fit_model(model, tr_X, tr_y):
    print('[*] Start to fit model')
    if EARLYSTOP:
        early_stopping = EarlyStopping(monitor='loss', patience=10)
    else:
        early_stopping = None

    hist = model.fit(tr_X, tr_y, validation_split=VALID_SPLIT, epochs=EPOCH_SIZE,
                     batch_size=BATCH_SIZE, shuffle=False, callbacks=early_stopping)
    print('[*] Fitting phase done\n')
    return model, hist


def test_model(model, te_X, te_y):
    print('[*] Start to test model')
    pred_y = np.argmax(model.predict(te_X, batch_size=BATCH_SIZE), axis=1)
    te_y = np.argmax(te_y, axis=1)
    print('[*] Testing phase done\n')
    rep = classification_report(te_y, pred_y, output_dict=True)
    rep.update({"accuracy": {"precision": None, "recall": None, "f1-score": rep["accuracy"], "support": rep['macro avg']['support']}})
    rep = pd.DataFrame(rep).transpose()
    return rep


def c_fit_and_result(X, model_func, method):
    print('[*] Start to fit model')
    model = None
     
    #Repeat the process for every number of clusters to find the optimal
    if (OPT_K == True):
        
        if method == 0:
            model = model_func(0)
            visualizer = KElbowVisualizer(model, k=K_RANGE, timings= True)
            visualizer.fit(X)
            visualizer.show()
            visualizer.show(outpath="image/kelbow_minibatchkmeans.png")
            print ("The optimal distance is", visualizer.elbow_score_, "using", visualizer.elbow_value_, "clusters")
            model = model_func(visualizer.elbow_value_)
            model.fit(X)
            
        else:
            wss = []
            n=1  
            best_model = None
            total = None
            for i in K_RANGE:
                model = model_func(i)
                model.fit(X)
                wss.append(model.inertia_) #!!! Square distance (only for kmeans!!!)
                if (i == 1):
                    total = model.inertia_
                    continue
                print ((wss[i-2] - wss[i-1]) / total)
                if ((wss[i-2] - wss[i-1]) / total > DIFF_VALUE):                 
                    best_model = model
                    n=i
            create_optimal_k_graph(wss, n-1)
            print ("The optimal distance is", wss[n-1], "using", n, "clusters")
            model = best_model
    else:
        model = model_func(N_CLUSTERS)
        model.fit(X)
        
    #print('Accuracy of the Model: ', metrics.accuracy_score(y, y_cluster))
    #print('Precision of the Model: ', metrics.precision_score(y, y_cluster))
    #print('Recall of the Model: ', metrics.recall_score(y, y_cluster))
    #print('F1-Score of the Model: ', metrics.f1_score(y, y_cluster))
    
    print('[*] Done!!! -> fit and result\n')
    return model


def kfold_cross_validation(model, X, y, kf):
    result = list()
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    for train, valid in kf.split(X, y):
        model.fit(X[train], y[train], epochs=EPOCH_SIZE, batch_size=BATCH_SIZE,
                  shuffle=False, callbacks=[early_stopping])
        temp = '%.4f' % (model.evaluate(X[valid], y[valid], batch_size=BATCH_SIZE)[1])
        result.append(temp)
    return result

def print_hyperprameter():
    print('[*] Hyperprameter')
    print('[-] Embedding Model:', EMBEDDING_MODEL)
    print('[-] Input Shape(Batch, TimeStep, Vector_Size):', BATCH_SIZE, SNIPPET_SIZE, EMBEDDING_DIM)
    print('[-] Output Shape:', CLASS_NUM)
    print('[-] Hiden layer Dimension:', HIDDEN_DIM)
    print('[-] Stateful:', STATEFUL)
    print('[-] Dropout rate:', DROPOUT_RATE)
    print('[-] Epoch size:', EPOCH_SIZE)
    print('[-] Split(test, validation):', TEST_SPLIT, VALID_SPLIT)
    print('')

def get_image_path():
    img_path = os.getcwd() + '/image/'
    if not isdir(img_path):
        os.mkdir(img_path)
    return img_path

def create_accuracy_graph(report, show=False):
    dirpath = get_image_path()
    title = report[0][2] + '_accuracy'
    filename = dirpath + datetime.datetime.now().strftime("%y%m%d_%H%M_") + str(report[0][1]) + '_' + str(report[0][0]) + '_' + title + '.png'
    accuracy = report[1]['history'].history['accuracy']
    val_accuracy = report[1]['history'].history['val_accuracy']
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title(title)
    plt.plot(accuracy, label='train')
    plt.plot(val_accuracy, label='test')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=100)
    if show: plt.show()
    plt.close()

def create_loss_graph(report, show=False):
    dirpath = get_image_path()
    title = report[0][2] + '_loss'
    filename = dirpath + datetime.datetime.now().strftime("%y%m%d_%H%M_") + str(report[0][1]) + '_' + str(report[0][0]) + '_' + title + '.png'
    loss = report[1]['history'].history['loss']
    val_loss = report[1]['history'].history['val_loss']
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title(title)
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='test')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=100)
    if show: plt.show()
    plt.close()

def create_compare_accuracy_graph(reports, show=False):
    dirpath = get_image_path()
    filename = dirpath + datetime.datetime.now().strftime("%y%m%d_%H%M_") + 'accuracy_compare.png'
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title('Compare model_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    for model_name, report in reports.items():
        accuracy = report['history'].history['accuracy']
        plt.plot(accuracy, label=model_name)
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=100)
    if show: plt.show()
    plt.close()
    
def create_compare_loss_graph(reports, show=False):
    dirpath = get_image_path()
    filename = dirpath + datetime.datetime.now().strftime("%y%m%d_%H%M_") + 'loss_compare.png'
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title('Compare model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    for model_name, report in reports.items():
        loss = report['history'].history['loss']
        plt.plot(loss, label=model_name)
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=100)
    if show: plt.show()
    plt.close()

def create_compare_methods_graph(reports, name, show=False):
    dirpath = get_image_path()
    filename = dirpath + datetime.datetime.now().strftime("%y%m%d_%H%M_") + 'methods_compare.png'
    plt.rcParams["figure.figsize"] = (12,6)
    plt.title('Compare accuracy between methods')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    for model_name, report in reports.items():
        if model_name[1] == name or name == 'all':
            accuracy = report['history'].history['accuracy']
            if model_name[1] == 'none': plt.plot(report['history'].history['accuracy'], label=model_name[2])
            elif model_name[0] == 0: plt.plot(accuracy, label=model_name[1]+' ('+model_name[2]+')')
            else: plt.plot(accuracy, label=str(model_name[0])+'-'+model_name[1]+' ('+model_name[2]+')')

    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=150)
    if show: plt.show()
    plt.close()
    
def create_compare_methods_graph_avg(reports, name, show=False):
    dirpath = get_image_path()
    filename = dirpath + datetime.datetime.now().strftime("%y%m%d_%H%M_") + 'methods_compare_avg.png'
    plt.rcParams["figure.figsize"] = (12,6)
    plt.title('Compare avg accuracy between methods')
    plt.ylabel('avg accuracy')
    plt.xlabel('epoch')
    
    algorithms = []
    types = []
    
    for key in reports.keys():
        algorithms.append(key[2])
        types.append(key[1])
        
    algorithms = np.unique(algorithms)
    types = np.unique(types)

    for alg in algorithms:
        for met in types:
            if met == name or name == 'all':
                accuracy = []
                for model_name, report in reports.items():
                    if met in model_name and alg in model_name:
                        if met == 'none': plt.plot(report['history'].history['accuracy'], label=model_name[2])
                        elif model_name[0] == 0: plt.plot(report['history'].history['accuracy'], label=model_name[1]+' ('+model_name[2]+')')
                        else: accuracy.append(report['history'].history['accuracy'])
                if met != 'none': plt.plot(np.mean( np.array(accuracy), axis=0 ), label='avg '+met+' ('+alg+')')

    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=150)
    if show: plt.show()
    plt.close()

def create_optimal_k_graph(wss, n, show=False):
    dirpath = get_image_path()
    filename = dirpath + datetime.datetime.now().strftime("%y%m%d_%H%M_") + 'optimal_k.png'
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title('Elbow Method For Optimal K')
    plt.ylabel('WSS')
    plt.xlabel('K')
    plt.plot(K_RANGE, wss, 'bx-', markevery=[n], label='K')
    plt.legend(loc='upper right')
    plt.savefig(filename, dpi=100)
    if show: plt.show()
    plt.close()

def autoenc(X, y):
    
    full_dim = SNIPPET_SIZE
    encoding_dim1 = full_dim
    encoding_dim2 = encoding_dim1/2
    encoding_dim3 = encoding_dim2/2

    encoder_input_data = Input(shape=(full_dim,))

    encoded_layer1 = Dense(encoding_dim1, activation='tanh')(encoder_input_data) #relu
    encoded_layer2 = Dense(encoding_dim2, activation='relu')(encoded_layer1)
    encoded_layer3 = Dense(encoding_dim3, activation='relu', name="ClusteringLayer")(encoded_layer2)
    encoder_model = Model(encoder_input_data, encoded_layer3)
    #encoder_model.compile(optimizer="RMSprop", loss=tf.keras.losses.mean_squared_error)
    
    decoded_layer3 = Dense(encoding_dim2, activation='tanh')(encoded_layer3)
    decoded_layer2 = Dense(encoding_dim1, activation='relu')(decoded_layer3)
    decoded_layer1 = Dense(full_dim, activation='relu')(decoded_layer2) #sigmoid
    
    adam = Adamax(lr=0.001)
    autoencoder_model = Model(encoder_input_data, outputs=decoded_layer1, name="Encoder")
    autoencoder_model.compile(optimizer=adam, loss='mse') #RMSprop
    #tf.keras.utils.plot_model(model=autoencoder_model, rankdir="LR", dpi=130, show_shapes=True, to_file="autoencoder.png")
    
    train_data, _, test_data, _ = _split_data(X, y, TEST_SPLIT, split_shuffle=True)
    history = autoencoder_model.fit(train_data, train_data, epochs=EPOCH_SIZE+50, batch_size=BATCH_SIZE, shuffle=True, validation_data=(test_data, test_data))
  
    encoded_groups = encoder_model.predict(X)
    #encoded_groups = encoder_model(X)
      
    return encoded_groups, history, encoder_model


def autoenc_emb(X, y, embedding_matrix):
    
    full_dim = SNIPPET_SIZE
    input_dim, output_dim = embedding_matrix.shape


    encoder_inputs = Input(shape=(full_dim,), name='Encoder-Input')
    emb_layer = add_embedding_layer(embedding_matrix)
    x = emb_layer(encoder_inputs)
    state_h = Bidirectional(LSTM(128, activation='relu', name='Encoder-Last-LSTM'))(x)
    encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
    seq2seq_encoder_out = encoder_model(encoder_inputs)
    
    decoded = RepeatVector(full_dim)(seq2seq_encoder_out)
    decoder_lstm = Bidirectional(LSTM(128, return_sequences=True, name='Decoder-LSTM-before'))
    decoder_lstm_output = decoder_lstm(decoded)
    decoder_dense = Dense(full_dim, activation='softmax', name='Final-Output-Dense-before', bias_initializer='zeros')
    decoder_outputs = decoder_dense(decoder_lstm_output)
    
    seq2seq_Model = Model(encoder_inputs, decoder_outputs)
    seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')
    history = seq2seq_Model.fit(X, np.expand_dims(X, -1), batch_size=BATCH_SIZE, epochs=10)
    
    encoded_groups = encoder_model.predict(X)
    
    return encoded_groups, history


def autoenc_3(X, y):
    n_inputs = X.shape[1]
    # split into train test sets
    X_train, y_test, X_test, y_test = _split_data(X, y, TEST_SPLIT, split_shuffle=True)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    
    if isfile('encoder.h5'): encoder = load_model('encoder.h5')
    else:
        # define encoder
        visible = Input(shape=(n_inputs,))
        # encoder level 1
        e = Dense(n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # encoder level 2
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # bottleneck
        n_bottleneck = n_inputs
        bottleneck = Dense(n_bottleneck)(e)
        
        # define decoder, level 1
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # decoder level 2
        d = Dense(n_inputs*2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # output layer
        output = Dense(n_inputs, activation='linear')(d)
        # define autoencoder model
        model = Model(inputs=visible, outputs=output)
        # compile autoencoder model
        model.compile(optimizer='adam', loss='mse')
        # plot the autoencoder
        plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
        # fit the autoencoder model to reconstruct input
        history = model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,X_test))
        # plot loss
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        # define an encoder model (without the decoder)
        encoder = Model(inputs=visible, outputs=bottleneck)
        plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
        # save the encoder to file
        encoder.save('encoder.h5')
    
    encoded_groups = encoder.predict(X)
    
    return encoded_groups, history


def create_Excel(reports, c, types):
    path = 'results\\Results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + '.xlsx'
    if isfile(path): os.remove(path)
    
    workbook = xls.Workbook(path)
    sheet = workbook.add_worksheet('Summary')
    data = workbook.add_worksheet('Data')
    sheet.set_column('A:M', 12)
    row = 1
    drow = 1
    
    #Row for each group
    for report in reports.items():
        n = report[0][0]
        if report[1]['result'] is not None:
            sheet.write_row(row, 0, [n, report[0][1], c[n] if report[0][1] == 'type' else 'none',
                                     report[1]['result']['support']['weighted avg'],
                                     report[1]['result']['support']['0']/report[1]['result']['support']['weighted avg'],
                                     report[1]['history'].history['accuracy'][-1],
                                     report[1]['result']['precision']['weighted avg'],
                                     report[1]['result']['recall']['weighted avg'],
                                     report[1]['result']['f1-score']['weighted avg'], 
                                     report[1]['history'].history['TP'][-1],
                                     report[1]['history'].history['TN'][-1],
                                     report[1]['history'].history['FP'][-1],
                                     report[1]['history'].history['FN'][-1]
                                     ])
        row += 1
        
        data.write_column(drow, 0, list(range(1,EPOCH_SIZE+1)))
        data.write_column(drow, 1, np.full(EPOCH_SIZE, n))
        data.write_column(drow, 2, np.full(EPOCH_SIZE, report[0][1]))
        data.write_column(drow, 3, np.full(EPOCH_SIZE, c[n] if report[0][1] == 'type' else 'none'))
        data.write_column(drow, 4, report[1]['history'].history['accuracy'])
        data.write_column(drow, 5, report[1]['history'].history['loss'])
        data.write_column(drow, 6, report[1]['history'].history['TP'])
        data.write_column(drow, 7, report[1]['history'].history['TN'])
        data.write_column(drow, 8, report[1]['history'].history['FP'])
        data.write_column(drow, 9, report[1]['history'].history['FN'])
        drow += EPOCH_SIZE
    
    #Make tables with the previous data
    sheet.add_table(0, 0, row-1, 12, {'name': 'Summary', 'first_column': True, 'columns': 
                                      [{'header': 'Group'}, {'header': 'Method'}, {'header': 'Type'}, {'header': 'Number'}, {'header': '%_Neutro'}, 
                                       {'header': 'Accuracy'}, {'header': 'Precision'}, {'header': 'Recall'}, {'header': 'F1-score'}, 
                                       {'header': 'TP'}, {'header': 'TN'}, {'header': 'FP'}, {'header': 'FN'}]})
 
    data.add_table(0, 0, drow-1, 9, {'name': 'Data', 'first_column': True, 'columns': 
                                     [{'header': 'Epoch'}, {'header': 'Group'}, {'header': 'Method'}, {'header': 'Type'}, {'header': 'Accuracy'}, 
                                      {'header': 'Loss'}, {'header': 'TP'}, {'header': 'TN'}, {'header': 'FP'}, {'header': 'FN'}]})
    #To check types per group
    if bool(types):
        sep = workbook.add_worksheet('Type')
        trow = 1
        for ty in types:
            n=0
            for group in types[ty]:
                #print( np.vectorize(c.get)(np.unique(types[ty])) )
                n += 1
                for u in np.unique(group):
                    numb = len(group[group == u])
                    sep.write_row(trow, 0, [n, ty, u, numb, numb/len(group)])
                    trow +=1
                    
        sep.add_table(0, 0, trow-1, 4, {'name': 'Type', 'first_column': True, 'columns': 
                                         [{'header': 'Group'}, {'header': 'Method'}, {'header': 'Type'}, {'header': 'Number'}, {'header': 'Percent'}]})
    
    workbook.close()
    
    
def removeMinNeutro(groups, labels, name, c, removed):
    n=0
    #X_new = np.ndarray(shape=(0, SNIPPET_SIZE), dtype='int32') #Only if X must be also reduced   
        
    for label in labels:
        vul = len(label[label.argmax(axis=1) == 0]) / len(label)
        if vul > MIN_NEUTRO:
            print('MIN_NEUTRO enabled: Removed group', n, 'with a', vul*100, '% of neutral code in', len(label), 'elements')
            removed[name, n] = [groups.pop(n), labels.pop(n)]
            if name == 'type' and c is not None: removed['names', n] = c[n]
        #else: X_new = np.concatenate((X_new, groups[n]), axis=0)
        n += 1
    return groups, labels #, X_new
        
    
def run(opt='training'):
    models = {
        # 'lstm': lstm, 'blstm': blstm, 'multi_lstm': multi_lstm, 'multi_lstm': multi_lstm,
        # 'gru': gru, 'bgru': bgru, 'multi_gru': multi_gru, 'multi_bgru': multi_bgru
        'BLSTM': blstm, 'BGRU': bgru
    }
    reports = dict()
    if opt=='training':
        (tr_X, tr_y), (te_X, te_y), embedding_matrix = load_snippet_data(EMBEDDING_MODEL)
        for model_name, model_func in models.items():
            temp = dict()
            model = model_func(embedding_matrix)
            result, history = fit_and_result(model, tr_X, tr_y, te_X, te_y)
            temp.update({'result': result, 'history': history})
            reports[0, model_name] = temp
    elif opt=='kfold':
        X, y, kf, embedding_matrix = load_snippet_data_for_kfold(EMBEDDING_MODEL)
        for model_name, model_func in models.items():
            temp = dict()
            model = model_func(embedding_matrix)
            result = kfold_cross_validation(model, X, y, kf)
            temp.update({'result': result})
            reports[0, model_name] = temp
    else:
        print('[!] Please check the opt name:', opt)
        return
    print_hyperprameter()
    for model_name in models.keys():
        print('[*]', model_name)
        print(reports[0, model_name]['result'])
        create_accuracy_graph(reports, 'none', model_name, 0)
        create_loss_graph(reports, 'none', model_name, 0)
    create_compare_accuracy_graph(reports)
    create_compare_loss_graph(reports)
          
def normal_run(name, reports, models, tr_X, tr_y, te_X, te_y, embedding_matrix):

    #(tr_X, tr_y), (te_X, te_y), embedding_matrix = load_snippet_data(EMBEDDING_MODEL)
    for model_name, model_func in models.items():
        temp = dict()
        model = model_func(embedding_matrix)
        result, history = fit_and_result(model, tr_X, tr_y, te_X, te_y)
        temp.update({'result': result, 'history': history})
        reports[0, name, model_name] = temp
    
    print_hyperprameter()

#Use type of vul to train the dataset
def type_c(X, y, t, types=None, encoded_groups=None):
    
    groups = npi.group_by(t).split(X)
    labels = npi.group_by(t).split(y)
    
    print ("\n[#] Dataset split in groups by type of vulnerability")
    return groups, labels, None

def kmeans_c(X, y, t, types, encoded_groups):
    
    #Get label of group classification and divide dataset
    if encoded_groups is None: model = c_fit_and_result(X, kmeans, 0)
    else: model = c_fit_and_result(encoded_groups, kmeans, 0)
  
    groups = npi.group_by(model.labels_).split(X)
    labels = npi.group_by(model.labels_).split(y)
    if not SME and t is not None:
        if encoded_groups is None: types['kmeans'] = npi.group_by(model.labels_).split(t)
        else: types['autoenc-kmeans'] = npi.group_by(model.labels_).split(t)
    
    return groups, labels, model
           
#Train with grouped dataset
def group_run(tipo, reports, models, groups, labels, embedding_matrix):

    train_set = {'samples': np.ndarray(shape=(0, SNIPPET_SIZE)), 'labels': np.ndarray(shape=(0, 2))} #'samples': [], 'labels': []
    test_set = {'samples': np.ndarray(shape=(0, SNIPPET_SIZE)), 'labels': np.ndarray(shape=(0, 2))} #'samples': [], 'labels': []
    
    n = 0
    for group, label in zip(groups, labels):
        n += 1
        
        print('\n[#] Starting group', n, 'with', len(group), 'samples\n')
        #dataset is separated (70% training - 30% test)
        tr_X, tr_y, te_X, te_y = _split_data(group, label, TEST_SPLIT, split_shuffle=True)        
        
        train_set['samples'] = np.concatenate((train_set['samples'], tr_X), axis=0)
        train_set['labels'] = np.concatenate((train_set['labels'], tr_y), axis=0)
        test_set['samples'] = np.concatenate((test_set['samples'], te_X), axis=0)
        test_set['labels'] = np.concatenate((test_set['labels'], te_y), axis=0)  

        print_hyperprameter()
        #A supervised model is used for each group to get optimal result
        for model_name, model_func in models.items():
            temp = dict()
            model = model_func(embedding_matrix)
            result, history = fit_and_result(model, tr_X, tr_y, te_X, te_y)
            temp.update({'result': result, 'history': history}) #'values':{'number': len(tr_X) + len(te_X), 'vul':(len(tr_y[tr_y.argmax(axis=1) == 0])+len(tr_y[tr_y.argmax(axis=1) == 0])) / (len(tr_y)+len(te_y))}}
            reports[n, tipo, model_name] = temp       
            #result.to_excel('results/'+str(n)+tipo+model_name+'.xlsx')
            #with open('results\\'+str(n)+tipo+model_name+'.txt', "w") as text_file:
            #    text_file.write(result)
    
    print ("\n[#] Group training finished!\n")
    return train_set['samples'], train_set['labels'], test_set['samples'], test_set['labels']

def alt_run_fit(name, aux, reports, s_models, groups, labels, embedding_matrix):
     
    n = 0
    #Training set groups are used to fit models
    for group, label in zip(groups, labels):
        n += 1
        print('\n[#] Starting group', n, 'with', len(group), 'samples\n')
    
        #A supervised model is trained for each group
        for model_name, model_func in s_models.items():
            model = model_func(embedding_matrix)
            model, history = fit_model(model, group, label)
            aux[n, name, model_name] = model
            temp = dict()
            temp.update({'result': None, 'history': history})
            reports[n, name, model_name] = temp
                   
    
def alt_run_result(name, aux, reports, s_models, te_X, te_y, encoded_test_groups):
    #Then test-set is classified in these groups to choose the model that will be tested ----------------------------
    n = 0
    c_model = aux [name]

    if encoded_test_groups is None: pred = c_model.predict(te_X)
    else: pred = c_model.predict(encoded_test_groups)
    
    groups = npi.group_by(pred).split(te_X)
    labels = npi.group_by(pred).split(te_y)
    
    for group, label in zip(groups, labels):
        n += 1
        print('\n[#] Starting group', n, 'with', len(group), 'samples\n')

        #A supervised model is used for each group to get optimal result
        for model_name, model_func in s_models.items():
            model = aux[n, name, model_name]
            result = test_model(model, group, label)
            reports[n, name, model_name].update({'result': result})

    
def start():
       
    methods = {    
        'type': type_c,
        'kmeans': kmeans_c,
    }
    
    #models that will be used for supervised training
    s_models = {
        # 'lstm': lstm, 'blstm': blstm, 'multi_lstm': multi_lstm, 'multi_lstm': multi_lstm,
        # 'gru': gru, 'bgru': bgru, 'multi_gru': multi_gru, 'multi_bgru': multi_bgru
        'BLSTM': blstm #, 'BGRU': bgru
    }
    
    if not isdir('results'): os.makedirs('results')
    reports = dict() #To save the summary results and history values
    aux = dict() #To save models which will be reused in the test set
    types = dict()
    removed = dict()
    c = None
    t = None
    
    #Dataset is loaded and transformed to input vector (y is label)
    if SME:
        X, y, embedding_matrix = alt_load_snippet_data()
    else:
        X, t, y, embedding_matrix = wo_sme_load_snippet_data() #without oversampling
        d = dict([(i,j+1) for j,i in enumerate(sorted(set(t)))])
        c = {v: k for k, v in d.items()}
        c[0] = 'none'
        #ty = [d[j] for j in t]
        
        #Deteling Resource Management group
        n = np.where(t == 'Resource Managment')
        X = np.delete(X, n, axis=0)
        y = np.delete(y, n, axis=0)
        t = np.delete(t, n)

    if not GROUP_FIRST:
        #dataset is separated (70% training - 30% test)
        tr_X, tr_y, te_X, te_y = _split_data(X, y, TEST_SPLIT, split_shuffle=True)
    
    #Execution of different methods ----------------------------------------------------------------------------------------------------

    if GROUP_FIRST:
        #For each method a normal run is done to compare results with same dataset
        for name, method in methods.items():
            if name == 'type' and SME: continue
        
            groups, labels, _ = method(X, y, t, types, None)
            if ONLY_VUL: groups, labels = removeMinNeutro(groups, labels, name, c, removed)
            tr_X, tr_y, te_X, te_y = group_run(name, reports, s_models, groups, labels, embedding_matrix)
            if NORMAL_RUN: normal_run(name, reports, s_models, tr_X, tr_y, te_X, te_y, embedding_matrix)
            
            create_compare_methods_graph(reports, name)
            create_compare_methods_graph_avg(reports, name)
        
        if AUTOENCOD: #All methods are done using encoded data for clustering
            encoded_groups, history, _ = autoenc(X,y)
            for name, method in methods.items():
                if name == 'type': continue
                name = 'autoenc-' + name
                
                groups, labels, _ = method(X, y, t, types, encoded_groups)
                if ONLY_VUL: groups, labels = removeMinNeutro(groups, labels, name, c, removed)
                tr_X, tr_y, te_X, te_y = group_run(name, reports, s_models, groups, labels, embedding_matrix)
                if NORMAL_RUN: normal_run(name, reports, s_models, tr_X, tr_y, te_X, te_y, embedding_matrix)
                
                create_compare_methods_graph(reports, name)
                create_compare_methods_graph_avg(reports, name)
           
    else: #Execution of alternative mode  
        #Dataset is split first, so groups are made with the training set only. Finally, test_set is grouped and used just to predict
        for name, method in methods.items():
            if name == 'type': continue
        
            groups, labels, c_model = method(tr_X, tr_y, t, types, None)
            # ONLY_VUL: groups, labels = removeMinNeutro(groups, labels, name, c, removed) #!!! Can't remove groups, the model is built
            aux[name] = c_model #Classif. model is saved to used it again in the test set
            alt_run_fit(name, aux, reports, s_models, groups, labels, embedding_matrix)
            alt_run_result(name, aux, reports, s_models, te_X, te_y, None)
            if NORMAL_RUN: normal_run(name, reports, s_models, tr_X, tr_y, te_X, te_y, embedding_matrix)
            
            create_compare_methods_graph(reports, name)
            create_compare_methods_graph_avg(reports, name)
        
        if AUTOENCOD: #All methods are done using encoded data for clustering
            encoded_groups, history, encoder_model = autoenc(tr_X, tr_y)
            for name, method in methods.items():
                if name == 'type': continue
                name = 'autoenc-' + name
                
                groups, labels, c_model = method(tr_X, tr_y, t, types, encoded_groups)
                aux[name] = c_model #Classif. model is saved to used it again in the test set
                alt_run_fit(name, aux, reports, s_models, groups, labels, embedding_matrix)
                encoded_test_groups = encoder_model.predict(te_X)
                alt_run_result(name, aux, reports, s_models, te_X, te_y, encoded_test_groups)
                if NORMAL_RUN: normal_run(name, reports, s_models, tr_X, tr_y, te_X, te_y, embedding_matrix)
                
                create_compare_methods_graph(reports, name)
                create_compare_methods_graph_avg(reports, name)
    #-----------------------------------------------------------------------------------------------------------------------------------

    #Normal execution with the whole dataset randomized
    tr_X, tr_y, te_X, te_y = _split_data(X, y, TEST_SPLIT, split_shuffle=True)
    normal_run('none', reports, s_models, tr_X, tr_y, te_X, te_y, embedding_matrix)

    #Print reports
    if GET_CHARTS:
        print ("\n[#] Printing reports")
        for report in reports.items():
            create_accuracy_graph(report)
            create_loss_graph(report)
        #create_compare_accuracy_graph(reports) #To compare supervised methods
        #create_compare_loss_graph(reports) #To compare supervised methods
        
    #Make a comparison graph
    create_compare_methods_graph(reports, 'all')
    create_compare_methods_graph_avg(reports, 'all')
    create_Excel(reports, c, types)


if __name__=='__main__':
    #opt='training'
    #run(opt)
    embed_opt='w2v'
    start()
    
