import os
import random
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import metrics

def c_sample_data(lstm_output, y_train, n_samples):
  vectors = [] # sample positive data
  for i in range(len(lstm_output)):
    if y_train[i, 1] ==1:
      vectors.append(lstm_output[i,:])
  index = random.sample(range(len(vectors)), n_samples)
  vectors = np.array(vectors)[index,:]
  return vectors
  
def c_get_generative(G_in, dense_dim=128*2, out_dim=128*2, lr=1e-3):
  def com_loss(G_in, G_out):
    def com_loss_fixed(y_true, y_pred):
      kl_loss = K.sum(K.square(G_in - G_out))
      xent_loss = K.binary_crossentropy(y_true, y_pred)
      return 0.05 * kl_loss + xent_loss

    return com_loss_fixed

  x = Dense(dense_dim)(G_in)
  x = LeakyReLU(0.2)(x)
  x = Dense(dense_dim)(x)
  x = LeakyReLU(0.2)(x)
  G_out = Dense(out_dim, activation='tanh')(x)
  G = Model(G_in, G_out)
  opt = SGD(lr=lr)
  G.compile(loss='binary_crossentropy', optimizer=opt)
  # G.compile(loss=com_loss(G_in=G_in, G_out=G_out), optimizer=opt)
  return G, G_out
  
def c_get_discriminative(D_in, dense_dim=128*2, out_dim=128*2, lr=1e-3, drate = .25, leak=.2):
  x = Dense(dense_dim)(D_in)
  x = LeakyReLU(leak)(x)
  x = Dropout(drate)(x)

  x = Dense(dense_dim)(x)
  x = LeakyReLU(leak)(x)
  x = Dropout(drate)(x)

  x = Dense(dense_dim)(x)
  x = LeakyReLU(leak)(x)
  D_out = Dense(out_dim, activation='sigmoid')(x)

  D = Model(D_in, D_out)
  dopt = Adam(lr=lr)
  D.compile(loss='binary_crossentropy', optimizer=dopt)
  return D, D_out
    
def c_set_trainability(model, trainable=False):
  model.trainable = trainable
  for layer in model.layers:
    layer.trainable = trainable
        
def c_make_gan(GAN_in, G, D):
  def com_loss(G_in, G_out):
    def com_loss_fixed(y_true, y_pred):
      kl_loss = K.sum(K.square(G_in - G_out))
      xent_loss = K.binary_crossentropy(y_true, y_pred)
      return 0.05 * kl_loss + xent_loss

    return com_loss_fixed

  set_trainability(D, False)
  x = G(GAN_in)
  GAN_out = D(x)
  GAN = Model(GAN_in, GAN_out)
  GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
  # GAN.compile(loss=com_loss(G_in=GAN_in, G_out=x), optimizer=G.optimizer)
  return GAN, GAN_out

def c_sample_data_and_gen(G, lstm_output, y_train, n_samples, noise_dim=128*2):
  XT = c_sample_data(lstm_output, y_train, n_samples=n_samples)
  # XN_noise = np.random.uniform(0, 1, size=[n_samples, noise_dim])
  XN_noise = [] # use negative data
  for i in range(len(lstm_output)):
    if y_train[i, 1] == 0:
      XN_noise.append(lstm_output[i,:])
  index = random.sample(range(len(XN_noise)), n_samples)
  XN_noise = np.array(XN_noise)[index,:]  
  
  XN = G.predict(XN_noise)
  X = np.concatenate((XT, XN))
  y = np.zeros((2 * n_samples, 2))
  y[:n_samples, 1] = 1
  y[n_samples:, 0] = 1

  return X, y
     
def c_pretrain(G, D, lstm_output, y_train, n_samples, noise_dim=128*2, batch_size=512):
  X, y = c_sample_data_and_gen(G, lstm_output, y_train, n_samples=n_samples, noise_dim=noise_dim)
  c_set_trainability(D, True)
  D.fit(X, y, epochs=1, batch_size=batch_size)

def c_sample_noise(G, n_samples, noise_dim=128*2):
  # X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
  X = [] # sample negative data
  for i in range(len(lstm_output)):
    if y_train[i, 1] ==0:
      X.append(lstm_output[i,:])
  index = random.sample(range(len(X)), n_samples)
  X = np.array(X)[index, :]
  
  y = np.zeros((n_samples, 2))
  y[:, 1] = 1

  return X, y
    
def c_train(GAN, G, D, lstm_output, y_train, n_samples, epochs=500, noise_dim=128*2, batch_size=512, verbose=True, v_freq=50):
  d_loss = []
  g_loss = []
  e_range = range(epochs)
  # if verbose:
  #   e_range = tqdm(e_range)
    
  for epoch in e_range:
    X, y = c_sample_data_and_gen(G, lstm_output, y_train, n_samples=n_samples, noise_dim=noise_dim) # train D
    c_set_trainability(D, True)
    d_loss.append(D.train_on_batch(X, y))
        
    X, y = c_sample_noise(G, n_samples=n_samples, noise_dim=noise_dim) # train G
    c_set_trainability(D, False)
    g_loss.append(GAN.train_on_batch(X, y))
        
    if verbose and (epoch + 1) % v_freq == 0:
      print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
        
  return d_loss, g_loss