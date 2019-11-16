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

def sample_data(lstm_output, y_train, n_samples):
  vectors = []
  for i in range(len(lstm_output)):
    if y_train[i, 1] ==1:
      vectors.append(lstm_output[i,:])
  index = random.sample(range(len(vectors)), n_samples)
  vectors = np.array(vectors)[index,:]
  return vectors
  
def get_generative(G_in, dense_dim, out_dim, lr=1e-3):
  x = Dense(dense_dim)(G_in)
  x = LeakyReLU(0.2)(x)
  x = Dense(dense_dim)(x)
  x = LeakyReLU(0.2)(x)
  G_out = Dense(out_dim, activation='tanh')(x)
  G = Model(G_in, G_out)
  opt = SGD(lr=lr)  
  G.compile(loss='binary_crossentropy', optimizer=opt)
  return G, G_out
  
def get_discriminative(D_in, dense_dim, out_dim, lr=1e-3, drate = .25, leak=.2):
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

def sample_data_and_gen(G, lstm_output, y_train, n_samples, noise_dim):
  XT = sample_data(lstm_output, y_train, n_samples=n_samples)
  XN_noise = np.random.uniform(0, 1, size=[n_samples, noise_dim])
  XN = G.predict(XN_noise)
  X = np.concatenate((XT, XN))
  y = np.zeros((2 * n_samples, 2))
  y[:n_samples, 1] = 1
  y[n_samples:, 0] = 1

  return X, y
     
def pretrain(G, D, lstm_output, y_train, n_samples, noise_dim, batch_size=512):
  X, y = sample_data_and_gen(G, lstm_output, y_train, n_samples=n_samples, noise_dim=noise_dim)
  set_trainability(D, True)
  D.fit(X, y, epochs=1, batch_size=batch_size)

def sample_noise(G, n_samples, noise_dim):
  X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
  y = np.zeros((n_samples, 2))
  y[:, 1] = 1

  return X, y
    
def train(GAN, G, D, lstm_output, y_train, n_samples, noise_dim, epochs=500, batch_size=512, verbose=True, v_freq=50):
  d_loss = []
  g_loss = []
  e_range = range(epochs)
  # if verbose:
  #   e_range = tqdm(e_range)
    
  for epoch in e_range:
    X, y = sample_data_and_gen(G, lstm_output, y_train, n_samples=n_samples, noise_dim=noise_dim) # train D
    set_trainability(D, True)
    d_loss.append(D.train_on_batch(X, y))
        
    X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim) # train G
    set_trainability(D, False)
    g_loss.append(GAN.train_on_batch(X, y))
        
    if verbose and (epoch + 1) % v_freq == 0:
      print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
        
  return d_loss, g_loss