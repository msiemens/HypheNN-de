from time import time

import util
from dataset import data_training
from model import model


start_time = time()
data = data_training()

print('Training model...')
model.fit(data[0], data[1], batch_size=512, epochs=50)
model.save('data/model.h5')
print('Done')
print('Time:', util.time_delta(time() - start_time))
