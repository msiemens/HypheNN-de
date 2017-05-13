from time import time

import util
from dataset import data_validation
from model import model


start_time = time()

model.load_weights('data/model.h5')
data = data_validation()

print('Validating model...')
result = model.evaluate(data[0], data[1])
print()
print('Done')
print('Result:', result)
print('Time:', util.time_delta(time() - start_time))