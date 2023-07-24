import tensorflow as tf

import dataset

print('Building model...')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(dataset.WINDOW_SIZE, dataset.n_vocab)))
model.add(tf.keras.layers.Dense(dataset.n_vocab, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

print('Done')
print()
