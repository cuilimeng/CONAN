from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.callbacks import *
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras import backend as K
import tensorflow as tf
import random
from RareDiseaseDetection.gan import *
from RareDiseaseDetection.cgan import *


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

df = pd.read_csv('RareDiseaseDetection/data/ipf_train_data_100.csv', usecols=['cohort', 'visits'])
all_texts = [c.replace("[","").replace("]","").replace(",","") for c in list(df['visits'])]
all_labels = [c.replace("scoring", "0").replace("positive", "1") for c in list(df['cohort'])]

df = pd.read_csv('RareDiseaseDetection/data/ipf_test_data.csv', usecols=['cohort', 'visits'])
test_texts = [c.replace("[","").replace("]","").replace(",","") for c in list(df['visits'])]
test_labels = [c.replace("scoring", "0").replace("positive", "1") for c in list(df['cohort'])]

MAX_SEQUENCE_LENGTH = sum(len(l.split()) for l in all_texts)/len(all_texts)
MAX_SEQUENCE_LENGTH = int(MAX_SEQUENCE_LENGTH)

print (MAX_SEQUENCE_LENGTH)

# Data distribution
"""
import matplotlib.pyplot as plt
list_p = []
list_s = []
for i in range(len(all_texts)):
  if all_labels[i]=="0":
    list_s.append(len(all_texts[i].split()))
  else:
    list_p.append(len(all_texts[i].split()))
plt.hist(list_s,bins=100,color='g',alpha=0.4,edgecolor='b')
plt.show()
"""

train_len = len(all_texts)
all_texts.extend(test_texts)
all_labels.extend(test_labels)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
# labels = np.argmax(labels, axis=1)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

X_train = data[:train_len]
X_test = data[train_len:]
y_train = labels[:train_len]
y_test = labels[train_len:]

# Validation
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

# Focal loss
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

# LSTM
def BuildLSTM():
  EMBEDDING_DIM = 128
  
  model = Sequential()
  model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, 
                      input_length=MAX_SEQUENCE_LENGTH, name = 'Embedding_1'))
  model.add(LSTM(EMBEDDING_DIM, name = 'LSTM_1'))
  model.add(Dense(2, activation='softmax', name = 'Dense_1'))
  model.compile(loss=[focal_loss(gamma=2., alpha=.25)],
                optimizer='rmsprop', metrics=['acc'])
  return model

# RNN
def BuildRNN():
  EMBEDDING_DIM = 128
  
  model = Sequential()
  model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, 
                      input_length=MAX_SEQUENCE_LENGTH, name = 'Embedding_2'))
  model.add(SimpleRNN(EMBEDDING_DIM, name = 'RNN_1'))
  model.add(Dense(2, activation='softmax', name = 'Dense_2'))
  model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='rmsprop', metrics=['acc'])
  return model

# Logistic Regression
def BuildLR():
  EMBEDDING_DIM = 128
  model = Sequential()
  model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, 
                      input_length=MAX_SEQUENCE_LENGTH, name = 'Embedding_3'))
  model.add(Flatten())
  model.add(Dense(2, input_dim = EMBEDDING_DIM, activation='softmax', name = 'Dense_3'))
  model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='rmsprop', metrics=['acc'])
  return model

# RETAIN
def cal_softmax(g):
  attn_g = K.softmax(g, axis=-1)
  print("attn_g.shape:" + str(attn_g.shape))
  return attn_g

def cal_tanh(h):
  attn_h = K.tanh(h)
  print("attn_h.shape:" + str(attn_h.shape))
  return attn_h

def reshape_sum(v):
  return K.sum(v, axis = 1)

def BuildRetain():
  EMBEDDING_DIM = 128
  
  sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  print("sentence_input.shape:" + str(sentence_input.shape))
  embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, 
                              input_length=MAX_SEQUENCE_LENGTH, name = 'Embedding_4')(sentence_input)
  print("embedding_layer.shape:" + str(embedding_layer.shape))
  g = GRU(EMBEDDING_DIM, dropout = 0.5)(embedding_layer)
  h = GRU(EMBEDDING_DIM, dropout = 0.5)(embedding_layer)
  
  g = Dense(1)(g)
  print("g.shape:" + str(g.shape))
  h = Dense(EMBEDDING_DIM)(h)
  print("h.shape:" + str(g.shape))
  attn_g = Lambda(cal_softmax)(g)
  attn_h = Lambda(cal_tanh)(h)
  c = Multiply()([attn_g, attn_h])
  c = Multiply()([c, embedding_layer])
  c = Lambda(reshape_sum)(c)
  
  print("c.shape:" + str(c.shape))
  
  preds = Dense(2, activation='softmax')(c)
  
  model = Model(sentence_input, preds)
  model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='adagrad', metrics=['acc'])
  return model

# MiME
def BuildMime():
  EMBEDDING_DIM = 128
  sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  print("sentence_input.shape:" + str(sentence_input.shape))
  embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, 
                              input_length=MAX_SEQUENCE_LENGTH, name = 'Embedding_5')(sentence_input)
  print("embedding_layer.shape:" + str(embedding_layer.shape))
  h = GRU(EMBEDDING_DIM, dropout = 0.2)(embedding_layer)
  
  preds = Dense(2, activation='softmax')(h)
  
  model = Model(sentence_input, preds)
  model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='adam', metrics=['acc'])
  return model

# Dipole
def BuildDipole():
  EMBEDDING_DIM = 128
  sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  print("sentence_input.shape:" + str(sentence_input.shape))
  embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, 
                              input_length=MAX_SEQUENCE_LENGTH, name = 'Embedding_5')(sentence_input)
  print("embedding_layer.shape:" + str(embedding_layer.shape))
  h = Bidirectional(GRU(EMBEDDING_DIM, dropout = 0.2, return_sequences = True))(embedding_layer)
  l_att = Dense(1,activation='tanh')(h)
  l_att = Flatten()(l_att)
  l_att = Activation('softmax')(l_att)
  l_att = RepeatVector(EMBEDDING_DIM*2)(l_att)
  l_att = Permute([2, 1])(l_att)
  l_att = Multiply()([h, l_att])
  print(l_att.shape)
  l_att = Lambda(lambda x: K.sum(x, axis=1))(l_att)
  preds = Dense(2, activation='softmax')(l_att)
  
  model = Model(sentence_input, preds)
  model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='adam', metrics=['acc'])
  return model

# sGAN
def BuildsGAN():
  EMBEDDING_DIM = 128
  
  model = Sequential()
  model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, 
                      input_length=MAX_SEQUENCE_LENGTH, name = 'Embedding_1'))
  model.add(LSTM(EMBEDDING_DIM, name = 'LSTM_2'))
  model.add(Dense(2, activation='softmax', name = 'Dense_4'))
  model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='rmsprop', metrics=['acc'])
  return model

def sample_data(n_samples=200):
  vectors = []
  for i in range(len(lstm_output)):
    if y_train[i, 1] ==1:
      vectors.append(lstm_output[i,:])
  index = random.sample(range(len(vectors)), n_samples)
  vectors = np.array(vectors)[index,:]
  return vectors
  
def get_generative(G_in, dense_dim=128, out_dim=128, lr=1e-3):
  x = Dense(dense_dim)(G_in)
  x = LeakyReLU(0.2)(x)
  x = Dense(dense_dim)(x)
  x = LeakyReLU(0.2)(x)
  G_out = Dense(out_dim, activation='tanh')(x)
  G = Model(G_in, G_out)
  opt = SGD(lr=lr)  
  G.compile(loss='binary_crossentropy', optimizer=opt)
  return G, G_out
  
def get_discriminative(D_in, dense_dim=128, out_dim=128, lr=1e-3, drate = .25, leak=.2):
  x = Dense(dense_dim)(D_in)
  x = LeakyReLU(leak)(x)
  x = Dropout(drate)(x)

  x = Dense(dense_dim)(x)
  x = LeakyReLU(leak)(x)
  x = Dropout(drate)(x)

  x = Dense(dense_dim)(x)
  x = LeakyReLU(leak)(x)
  D_out = Dense(2, activation='sigmoid')(x)
  D = Model(D_in, D_out)
  dopt = Adam(lr=lr)
  D.compile(loss='binary_crossentropy', optimizer=dopt)
  return D, D_out
    
def set_trainability(model, trainable=False):
  model.trainable = trainable
  for layer in model.layers:
    layer.trainable = trainable
        
def make_gan(GAN_in, G, D):
  set_trainability(D, False)
  x = G(GAN_in)
  GAN_out = D(x)
  GAN = Model(GAN_in, GAN_out)
  GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
  return GAN, GAN_out

def sample_data_and_gen(G, noise_dim=128, n_samples=200):
  XT = sample_data(n_samples=n_samples)
  XN_noise = np.random.uniform(0, 1, size=[n_samples, noise_dim])
  XN = G.predict(XN_noise)
  X = np.concatenate((XT, XN))
  y = np.zeros((2 * n_samples, 2))
  y[:n_samples, 1] = 1
  y[n_samples:, 0] = 1

  return X, y
     
def pretrain(G, D, noise_dim=128, n_samples=200, batch_size=4096):
  X, y = sample_data_and_gen(G, noise_dim=noise_dim, n_samples=n_samples)
  set_trainability(D, True)
  D.fit(X, y, epochs=1, batch_size=batch_size)

def sample_noise(G, noise_dim=128, n_samples=200):
  X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
  y = np.zeros((n_samples, 2))
  y[:, 1] = 1

  return X, y
    
def train(GAN, G, D, epochs=100, n_samples=200, noise_dim=128, batch_size=4096, verbose=True, v_freq=50):
  d_loss = []
  g_loss = []
  e_range = range(epochs)
  # if verbose:
  #   e_range = tqdm(e_range)
    
  for epoch in e_range:
    X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim) # train D
    set_trainability(D, True)
    d_loss.append(D.train_on_batch(X, y))
        
    X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim) # train G
    set_trainability(D, False)
    g_loss.append(GAN.train_on_batch(X, y))
        
    if verbose and (epoch + 1) % v_freq == 0:
      print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
        
  return d_loss, g_loss

### Option 1: Build sGAN
model = BuildsGAN()
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=1024)

lstm_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('LSTM_2').output)
lstm_output = lstm_layer_model.predict(X_train, batch_size=1024)
print(lstm_output.shape)

lstm_test_output = lstm_layer_model.predict(X_test, batch_size=1024)
print(lstm_test_output.shape)

G_in = Input(shape=[128])
G, G_out = get_generative(G_in)
G.summary()

D_in = Input(shape=[128])
D, D_out = get_discriminative(D_in)
D.summary()

GAN_in = Input([128])
GAN, GAN_out = make_gan(GAN_in, G, D)
GAN.summary()

n_samples = 200

pretrain(G, D, lstm_output, y_train, n_samples)

# d_loss, g_loss = train(GAN, G, D, verbose=True)
d_loss, g_loss = train(GAN, G, D, lstm_output, y_train, n_samples, verbose=True)

# data_and_gen, _ = sample_data_and_gen(G, n_samples=200)
data_and_gen, _ = sample_data_and_gen(G, lstm_output, y_train, n_samples)


X_train = np.concatenate((lstm_output, data_and_gen))
new_y_train = []
for i in range(n_samples*2): new_y_train.append([0,1])
y_train = np.concatenate((y_train, np.array(new_y_train)))

model = Sequential()
model.add(Dense(2, activation='softmax', name = 'Dense_5'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.fit(X_train, y_train, epochs=2, batch_size=1024)
X_test = lstm_test_output
### End build sGAN

### Option 2: Build LR, RNN, LSTM, Mime, Retain, Dipole
model = BuildLSTM()
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=1024)
### End build other models

# predict probabilities for test set
yhat_probs = model.predict(X_test, batch_size=1024)
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
# fpr, tpr, thresholds = roc_curve(y_classes, yhat_probs, pos_label=1)
# print("ROC AUC: ", auc(fpr, tpr))
# PR AUC
precision, recall, thresholds = precision_recall_curve(y_classes, yhat_probs, pos_label=1)
print("PR AUC: ", auc(recall, precision))
# confusion matrix
matrix = confusion_matrix(y_classes, yhat_classes)
print(matrix)

