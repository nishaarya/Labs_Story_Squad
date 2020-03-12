#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install keras


# In[3]:


pip install tensorflow


# In[4]:


f = open(r'C:\Users\PC\Desktop\wget\mini_classes.txt')
#And for reading use
classes = f.readlines()
f.close()


# In[5]:


classes = [c.replace('\n','').replace(' ','_') for c in classes]


# In[6]:


#Download the dataset
#Loop over the classes and download the correspondent data
get_ipython().system('mkdir data')


# In[7]:


import urllib.request
def download():
  base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
  for c in classes:
    cls_url = c.replace('_', '%20')
    path = base+cls_url+'.npy'
    print(path)
    urllib.request.urlretrieve(path, 'data/'+c+'.npy')


# In[8]:


download()


# In[9]:


#Import Libraries

import os
import glob
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf


# In[10]:


def load_data(root, vfold_ratio=0.2, max_items_per_class= 5000 ):
    all_files = glob.glob(os.path.join(root, '*.npy'))
    #initialize variables 
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []
    #load each data file 
    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)
        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)
        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)
    data = None
    labels = None
    #randomize the dataset 
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]
    #separate into training and testing 
    vfold_size = int(x.shape[0]/100*(vfold_ratio*100))
    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]
    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names


# In[11]:


x_train, y_train, x_test, y_test, class_names = load_data('data')
num_classes = len(class_names)
image_size = 28


# In[12]:


# number of observations

print(len(x_train), 'train sequences')


# In[13]:


print(len(x_test), 'test sequences')


# In[14]:


import matplotlib.pyplot as plt
from random import randint
get_ipython().run_line_magic('matplotlib', 'inline')
idx = randint(0, len(x_train))
plt.imshow(x_train[idx].reshape(28,28)) 
print(class_names[int(y_train[idx].item())])


# In[15]:


# Reshape and normalize
x_train = x_train.reshape(x_train.shape[0], 784, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 784, 1).astype('float32')

x_train /= 255.0
x_test /= 255.0


# In[16]:


#reshapes y variables
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

y_train.shape, y_test.shape


# In[17]:


y_train.shape


# In[18]:


y_test.shape


# In[19]:


#x_train


# In[20]:


#The Model

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import BatchNormalization, LSTM, Conv1D, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


# In[27]:


print('Build model...')
model = Sequential()
model.add(LSTM(128, dropout=0.0, recurrent_dropout=0.2,input_shape = x_train.shape[1:]))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1, activation='softmax'))
# try using different optimizers and different optimizer configs
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
#model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=2,epochs=100)


# In[28]:


print("Train...")
model.fit(x_train, y_train, batch_size=45, nb_epoch=2,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=45)
print('Test score:', score)
print('Test accuracy:', acc)


# In[ ]:




