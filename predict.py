import sys

import numpy as np

import dataset
from model import model


def predict(word):
	model.load_weights('data/model.h5')

	windows = dataset.process_word(word.lower(), training=False)
	hyphenated = word[:2]

	for offset, window in enumerate(windows):
		result = model.predict(np.array([window]))

		#print('>>> {}{}{} => {: >5.2f} %'.format(
		#	word[:2 + offset],
		#	dataset.HYPHENATION_INDICATOR,
		#	word[2 + offset:],
		#	result[0][0] * 100
		#))

		if result[0][0] > 0.5:
			hyphenated += dataset.HYPHENATION_INDICATOR

		hyphenated += word[offset + 2]

	hyphenated += word[-1:]
	return hyphenated


if __name__ == '__main__':
	word = sys.argv[1]
	prediction = predict(word)
	print('Input:', word)
	print('Hyphenation:', prediction)