#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import randint
from numpy.random import rand
from numpy import zeros
from numpy import ones
from matplotlib import pyplot
import random
from numpy import vstack
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Reshape


#Input dimension generator
input_dim = 100


# In[2]:



# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

def define_discriminator(in_shape=(64,64,3)):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, (3,3), padding='same', data_format="channels_last", strides=(2, 2), input_shape=in_shape , activation=LeakyReLU(alpha=0.2)))
    discriminator.add(Dropout(0.4))
    discriminator.add(Conv2D(64, (3,3),padding='same', data_format="channels_last", strides=(2, 2), activation=LeakyReLU(alpha=0.2)))
    discriminator.add(Dropout(0.4))
    discriminator.add(Conv2D(64, (3,3),padding='same', data_format="channels_last", strides=(2, 2), activation=LeakyReLU(alpha=0.2)))
    discriminator.add(Dropout(0.4))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))  
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return discriminator


# In[5]:


discriminator = define_discriminator()
discriminator.summary()


# In[7]:


def define_generator(latent_dim):
    model = Sequential()
    # foundation for 8x8 image
    n_nodes = 128 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 128)))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsampling to 64x64
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (7,7), activation='tanh', padding='same'))
    return model


# In[8]:


generator = define_generator(100)
generator.summary()


# In[9]:


#GAN-model
discriminator.trainable = False
inputs = Input(shape=(input_dim, ))
hidden = generator(inputs)
output = discriminator(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
gan.summary()


# In[7]:


# create a data generator
datagen = ImageDataGenerator()

x_train = datagen.flow_from_directory('facedata'
                                       ,class_mode=None, color_mode = 'rgb',target_size=(64, 64), batch_size=768)


# In[8]:



def newImages(x_train=x_train):
    x = next(x_train)
    for i in range(768):
        x[i] = x[i].astype('float32')
        x[i] = x[i] / 127.5 - 1
    return x


# In[10]:


#Showing the images
x= newImages()
pyplot.figure(figsize=(8, 8))
for i in range(64):
    pyplot.subplot(10, 10, 1 + i)
    pyplot.axis('off')
    image = x[i]
    image = (image +1) /2
    plt.imshow(image)
plt.show()


# In[11]:



# select real samples
def generate_real_samples(x_train, n_samples):
    # choose random instances
    ix = randint(0, x_train.shape[0], n_samples)
    # retrieve selected images
    X = x_train[ix]
    # generate 'real' class labels (1)
    X.reshape(n_samples, 64,64,3)
    y = ones((n_samples, 1))
    return X, y
   
# generate n noise samples with class labels
def generate_noise_samples(n_samples):
    # generate uniform random numbers in [0,1]
    X = rand(4096* 3 * n_samples)
    # reshape into a batch of grayscale images
    X = X.reshape((n_samples, 4096*3))
    X = X.reshape((n_samples, 64, 64, 3))
    # generate 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

    # generate points in latent space as input for the generator
def generate_latent_points(input_dim, n_samples):
    # generate points in the latent space
    x_input = randn(input_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, input_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return x, y


# In[12]:



def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    x_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))


# In[13]:


#Train discriminator
def train_discriminator(model, number_iteration, batch_size, x_train):
    half_batch = int(batch_size/2)
    for i in range(number_iteration):
        #x_train = newImages()
        #Sample from real images
        x_real, y_real = generate_real_samples(x_train, half_batch)
        #Sample from noise
        x_noise, y_noise = generate_noise_samples((half_batch))
        #Train with real images
        _, real_acc = model.train_on_batch(x_real, y_real)
        #Train with noise
        _, noise_acc = model.train_on_batch(x_noise, y_noise)
        print('>%d real=%.0f%% noise=%.0f%%' % (i+1, real_acc*100, noise_acc*100))


# In[14]:


def train_gan(generator, discriminator, gan, input_dim, n_epochs, batch_size):
    x_train = newImages()
    batch_per_epoch = int(169396 / x_train.shape[0])-1
    subBatch_per_batch = int(x_train.shape[0]/batch_size)
    half_batch = int(batch_size/2)
    for i in range(n_epochs):
        datagen = ImageDataGenerator()
        x_train = datagen.flow_from_directory('facedata'
                                              ,class_mode=None, color_mode = 'rgb',target_size=(64, 64), batch_size=768)
        for j in range(batch_per_epoch):
            x_train = newImages()
            for k in range(subBatch_per_batch):
                #Trains 2 epochs per subset, in this case it will run 2 epochs on the 1000 images subset generated from newImages()
                #for k in range(2):
                x_real, y_real = generate_real_samples(x_train, half_batch)
                x_fake, y_fake = generate_fake_samples(generator, input_dim, half_batch)
                x, y = vstack((x_real, x_fake)), vstack((y_real, y_fake))
                discriminator_loss = discriminator.train_on_batch(x, y)
                x_gan = generate_latent_points(input_dim, batch_size)
                y_gan = ones((batch_size, 1))
                generator_loss = gan.train_on_batch(x_gan, y_gan)
                print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, discriminator_loss[0], generator_loss[0]))
            #if (j==1):
              #summarize_performance(i, generator, discriminator, x_train, input_dim)


# In[79]:


train_discriminator(discriminator, 100, 256, x)

#15ggr


# In[ ]:


train_gan(generator,discriminator,gan,input_dim, 20, 256)


# In[ ]:


train_gan(generator,discriminator,gan,input_dim, 20, 256)


# In[1]:


#Plot result
n_samples = 25
latent_points = generate_latent_points(input_dim, n_samples)
x = generator.predict(latent_points)
x = (x + 1) / 2.0
#print(x.shape)
for i in range(n_samples):
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    image = x[i]
    #print(image)
    #print(image)
    pyplot.imshow(image)
    
pyplot.show()


# In[ ]:




