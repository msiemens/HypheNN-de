import math

import keras as k
import tensorflow as tf

import dataset

print('Building model...')

model = k.models.Sequential()
model.add(k.layers.Flatten(input_shape=(dataset.WINDOW_SIZE, dataset.n_vocab)))
model.add(k.layers.Dense(dataset.n_vocab, activation='relu'))
model.add(k.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

print('Done')
print()