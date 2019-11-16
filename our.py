# coding=utf-8
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
import tensorflow as tf

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import *
from keras.models import Model, load_model, Sequential
from keras.callbacks import *
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import optimizers
from RareDiseaseDetection.gan import *
from RareDiseaseDetection.cgan import *

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []
        self.val_acc = []
        self.val_kappa = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict_onehot = (
            np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ_onehot = self.validation_data[1]
        val_predict = np.argmax(val_predict_onehot, axis=1)
        val_targ = np.argmax(val_targ_onehot, axis=1)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_auc = roc_auc_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        _val_kappa = cohen_kappa_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auc.append(_val_auc)
        self.val_acc.append(_val_acc)
        self.val_kappa.append(_val_kappa)
        print("Epoch: %d - val_accuracy: % f - val_precision: % f - val_recall % f val_f1: %f auc: %f kappa: %f" % (epoch, _val_acc, _val_precision, _val_recall, _val_f1, _val_auc, _val_kappa))
        return
      
MAX_SENT_LENGTH = 10
"""
241
798
"""
MAX_SENTS = 500
# MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 128
VALIDATION_SPLIT = 0.2

def clean_str(string):
  string = re.sub(r",", "", string)
  string = re.sub(r"\[\[", "", string)
  string = re.sub(r"\]\]", "", string)
  return string.strip()

data_train = pd.read_csv('RareDiseaseDetection/data/ipf_train_data_10.csv')
train_len = len(data_train)

data_test = pd.read_csv('RareDiseaseDetection/data/ipf_test_data.csv')
test_len = len(data_test)

codes = []
labels = []
visits = []

# change "visits" to "code_array", and "scoring" to "negative" when using NASH and IBD datasets
for idx in range(data_train.visits.shape[0]):
    visit = data_train.visits[idx]
    visit = clean_str(visit)
    diag = visit.split('] [')
    tmp = re.sub(r"\]\s\[", " ", visit)
    # if len(tmp)<500: continue
    visits.append(tmp)
    codes.append(diag)

    labels.append(data_train.cohort[idx].replace("scoring", "0").replace("positive", "1"))

train_len = len(labels)

for idx in range(data_test.visits.shape[0]):
    visit = data_test.visits[idx]
    visit = clean_str(visit)
    diag = visit.split('] [')
    # if len(tmp)<100: continue
    visits.append(re.sub(r"\]\s\[", " ", visit))
    codes.append(diag)

    labels.append(data_test.cohort[idx].replace("scoring", "0").replace("positive", "1"))

    
test_len = len(labels) - train_len

tokenizer = Tokenizer()
tokenizer.fit_on_texts(visits)

data = np.zeros((len(visits), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, diag in enumerate(codes):
  for j, d in enumerate(diag):
    if j < MAX_SENTS:
      wordTokens = text_to_word_sequence(d)
      k = 0
      for _, word in enumerate(wordTokens):
        if k < MAX_SENT_LENGTH:
          data[i, j, k] = tokenizer.word_index[word]
          k = k + 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

X_train = data[:train_len]
X_test = data[train_len:]
y_train = labels[:train_len]
y_test = labels[train_len:]


print('Number of positive and negative reviews in traing and validation set')
print(y_train.sum(axis=0))
print(y_test.sum(axis=0))

"""
GLOVE_DIR = "."
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
"""

"""
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
"""

class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class Transformer(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Transformer, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Transformer, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        # linear mapping
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        # inner product
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        # mask
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        # softmax
        A = K.softmax(A)
        # mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def focal_loss(gamma=2., alpha=.25):
  def focal_loss_fixed(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    
    epsilon = K.epsilon()
    
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
  return focal_loss_fixed


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            # weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)
                            # mask_zero=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
print(embedded_sequences.shape)
if MAX_SENTS==1:
  l_lstm = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True))(embedded_sequences)
  l_att = AttLayer(EMBEDDING_DIM)(l_lstm)
  print(l_att.shape)
  preds = Dense(2, activation='softmax')(l_att)
  model = Model(sentence_input, preds)
else:
  l_att = Transformer(3,64)([embedded_sequences,embedded_sequences,embedded_sequences])
  l_att = GlobalAveragePooling1D()(l_att)
  sentEncoder = Model(sentence_input, l_att)
  
  review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
  print(review_input.shape)
  review_encoder = TimeDistributed(sentEncoder)(review_input)
  print(review_encoder.shape)
  
  l_lstm_sent = Bidirectional(GRU(EMBEDDING_DIM, return_sequences=True))(review_encoder)
  print(l_lstm_sent.shape)
  l_att_sent = AttLayer(EMBEDDING_DIM, name = 'Att')(l_lstm_sent)
  print(l_att_sent.shape)

  preds = Dense(2, activation='softmax')(l_att_sent)
  model = Model(review_input, preds)

# Our w/o GAN
model.summary()
rmsprop = optimizers.RMSprop()
model.compile(loss=[focal_loss(gamma=2., alpha=.25)],
              optimizer=rmsprop,
              metrics=['acc'])
# m = Metrics()
# callbacks = [m]
if MAX_SENTS==1:
  X_train = np.squeeze(X_train, axis=1)
  X_test = np.squeeze(X_test, axis=1)
model.fit(X_train, y_train, epochs=2, batch_size=512)




### Option 1: Our w GAN
lstm_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Att').output)
lstm_output = lstm_layer_model.predict(X_train, batch_size=512)
print(lstm_output.shape)

lstm_test_output = lstm_layer_model.predict(X_test, batch_size=512)
print(lstm_test_output.shape)

G_in = Input(shape=[EMBEDDING_DIM * 2])
G, G_out = get_generative(G_in, dense_dim=EMBEDDING_DIM * 2, out_dim=EMBEDDING_DIM * 2)
G.summary()

D_in = Input(shape=[EMBEDDING_DIM * 2])
D, D_out = get_discriminative(D_in, dense_dim=EMBEDDING_DIM * 2, out_dim=EMBEDDING_DIM * 2)
D.summary()

GAN_in = Input([EMBEDDING_DIM * 2])
GAN, GAN_out = make_gan(GAN_in, G, D)
GAN.summary()

n_samples = 400
noise_dim = EMBEDDING_DIM * 2


pretrain(G, D, lstm_output, y_train, n_samples, noise_dim)

d_loss, g_loss = train(GAN, G, D, lstm_output, y_train, n_samples, noise_dim, verbose=True)

data_and_gen, _ = sample_data_and_gen(G, lstm_output, y_train, n_samples, noise_dim)

X_train = np.concatenate((lstm_output, data_and_gen))
new_y_train = []
for i in range(n_samples*2): new_y_train.append([0,1])
y_train = np.concatenate((y_train, np.array(new_y_train)))

model = Sequential()
model.add(Dense(2, activation='softmax', name = 'Dense_5'))
model.compile(loss=[focal_loss(gamma=2., alpha=.25)], optimizer='rmsprop', metrics=['acc'])

model.fit(X_train, y_train, epochs=5, batch_size=1024)
X_test = lstm_test_output

"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
random.seed(45)
index = random.sample(range(len(lstm_output)), 10000)
X = lstm_output[index,:]
X = np.concatenate((X, np.array(data_and_gen)[n_samples:,].astype(int)))
Y = y_train[index,1]
for i in range(n_samples): Y = np.concatenate((Y, [2]))
X_embed = TSNE(n_components=2).fit_transform(X)
red = Y == 1
green = Y == 0
blue = Y == 2
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 6))
plt.scatter(X_embed[green, 0], X_embed[green, 1], c="g"); plt.scatter(X_embed[red, 0], X_embed[red, 1], c="r"); plt.scatter(X_embed[blue, 0], X_embed[blue, 1], c="b")
plt.legend()
plt.axis('tight')
"""
###

### Option 2: Our w cGAN
lstm_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Att').output)
lstm_output = lstm_layer_model.predict(X_train, batch_size=512)
print(lstm_output.shape)

lstm_test_output = lstm_layer_model.predict(X_test, batch_size=512)
print(lstm_test_output.shape)

G_in = Input(shape=[EMBEDDING_DIM * 2])
G, G_out = c_get_generative(G_in, dense_dim= 128*2, out_dim= 128*2)
G.summary()

D_in = Input(shape=[EMBEDDING_DIM * 2])
D, D_out = c_get_discriminative(D_in, dense_dim= 128*2, out_dim= 128*2)
D.summary()

GAN_in = Input([EMBEDDING_DIM * 2])
GAN, GAN_out = c_make_gan(GAN_in, G, D)
GAN.summary()

n_samples = 400
noise_dim = 128*2

c_pretrain(G, D, lstm_output, y_train, n_samples, noise_dim)

d_loss, g_loss = c_train(GAN, G, D, lstm_output, y_train, n_samples, noise_dim, verbose=True)

data_and_gen, _ = c_sample_data_and_gen(G, lstm_output, y_train, n_samples, noise_dim)

X_train = np.concatenate((lstm_output, data_and_gen))
new_y_train = []
for i in range(n_samples*2): new_y_train.append([1,0])
y_train = np.concatenate((y_train, np.array(new_y_train)))

model = Sequential()
model.add(Dense(2, activation='softmax', name = 'Dense_5'))
model.compile(loss=[focal_loss(gamma=2., alpha=.25)], optimizer='rmsprop', metrics=['acc'])

model.fit(X_train, y_train, epochs=10, batch_size=1024)
X_test = lstm_test_output

"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
random.seed(45)
index = random.sample(range(len(lstm_output)), 10000)
X = lstm_output[index,:]
X = np.concatenate((X, np.array(data_and_gen)[n_samples:,].astype(int)))
Y = y_train[index,1]
for i in range(n_samples): Y = np.concatenate((Y, [2]))
X_embed = TSNE(n_components=2).fit_transform(X)
red = Y == 1
green = Y == 0
blue = Y == 2
colors = (0,0,0)
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 6))
plt.scatter(X_embed[green, 0], X_embed[green, 1], s=1, c="g"); plt.scatter(X_embed[red, 0], X_embed[red, 1], s=1, c="r"); plt.scatter(X_embed[blue, 0], X_embed[blue, 1], c="b")
plt.legend()
plt.axis('tight')

tmpX = X_embed
np.savetxt('tmpX.txt', tmpX, fmt='%f')
tmpY = Y
np.savetxt('tmpY.txt', tmpY, fmt='%f')
"""
###

# predict probabilities for test set
yhat_probs = model.predict(X_test, batch_size=512)
# predict crisp classes for test set
yhat_classes = np.argmax(yhat_probs, axis=1)
yhat_probs = yhat_probs[:,1]
y_classes = np.argmax(y_test, axis=1)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_classes, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_classes, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_classes, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_classes, yhat_classes)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(y_classes, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
fpr, tpr, thresholds = roc_curve(y_classes, yhat_probs, pos_label=1)
print("ROC AUC: ", auc(fpr, tpr))
# PR AUC
precision, recall, thresholds = precision_recall_curve(y_classes, yhat_probs, pos_label=1)
print("PR AUC: ", auc(recall, precision))
# confusion matrix
matrix = confusion_matrix(y_classes, yhat_classes)
print(matrix)
