import math

import numpy as np

DATA_CACHE_TRAINING = 'wordlist-training.npz'
DATA_CACHE_VALIDATION = 'wordlist-valdiation.npz'
WINDOW_SIZE = 8
TRAINING_SET = 150000
# VALIDATION_SET = 150000
HYPHENATION_INDICATOR = '·'

raw_text = open('wordlist.txt').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text) - set(['\n', HYPHENATION_INDICATOR]) | set([''])))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {v: k for k, v in char_to_int.items()}
#int_to_char[0] = '@'

n_chars = len(raw_text)
n_vocab = len(chars)

words = raw_text.splitlines()
words_training = words[:TRAINING_SET]
words_validation = words[TRAINING_SET:]


def process_data(words):
	X = []
	y = []

	for i, word in enumerate(words):
		data = process_word(word)
		X.extend(data[0])
		y.extend(data[1])

		if (i + 1) % 100 == 0:
			print('\rProcessed {} entries ({} %)'.format(i + 1, round((i + 1) / len(words) * 100)), end='')

	print()
	return np.asarray(X), np.asarray(y)


def process_word(word, training=True):
	X = []
	y = []

	word_int = [char_to_int[c] for c in word if c != HYPHENATION_INDICATOR]
	word_int = np.array([0, 0] + word_int + [0, 0])
	hyphenations = 0
	padded = False

	#print()
	#print()
	#print('>>> word:', word)
	#print('>>> word_int:', word_int)

	# Fill with zeros if needed
	if len(word_int) < WINDOW_SIZE:
		zeros = np.zeros(WINDOW_SIZE - len(word_int), dtype=word_int.dtype)
		word_int = np.concatenate((word_int, zeros))
		padded = True

	# Calculate the number of times we have to slide the window to cover
	# the whole word
	num_windows = len(word_int) - WINDOW_SIZE + 1
	indexer = np.arange(WINDOW_SIZE)[None, :] + np.arange(num_windows)[:, None]
	windows = word_int[indexer]

	#print('>>> num_windows:', num_windows)

	#_c = word[:2]

	# Calculate hyphenation positions
	for offset, window in enumerate(windows):
		#_w = ''.join([int_to_char[c] for c in window])
		#print('>>> offset:', offset)
		#print('>>> window:', window)
		#print('>>> word:', _w[0:4], _w[4:])

		o = offset + 2 + hyphenations  # + 2, 1x to check next char, 1x for ???
		#print('>>> o:', o)

		if training:
			hyphenation = word[o] == HYPHENATION_INDICATOR
			if hyphenation:
				hyphenations += 1
				#_c += HYPHENATION_INDICATOR

		#_c += _w[4]

		#print('>>> c:', word[o])
		#print('>>> _c:', _c, _w[4])
		#print('>>> h:', hyphenation)
		#print()

		one_hot = np.zeros((WINDOW_SIZE, n_vocab), dtype=np.bool)
		one_hot[np.arange(WINDOW_SIZE), window] = True

		# print(one_hot)
		#if not padded:
		#	_c += word[-1:]

		X.append(one_hot)
		if training:
			y.append(hyphenation)

	#if _c != word:
	#	print('>>> Error:')
	#	print('>>> Calculated:', _c)
	#	print('>>> Original:  ', word)
	# assert _c == word

	if training:
		return X, y
	else:
		return X


def data_training():
	try:
		data = np.load(DATA_CACHE_TRAINING)
		return data['X'], data['y']
	except FileNotFoundError:
		print('Preparing training data...')
		X, y = process_data(words_training)
		print('Storing data...')
		np.savez(DATA_CACHE_TRAINING, X=X, y=y)
		print('Done')
		print()

		return X, y


def data_validation():
	try:
		data = np.load(DATA_CACHE_TRAINING)
		return data['X'], data['y']
	except FileNotFoundError:
		print('Preparing validation data...')
		X, y = process_data(words_validation)
		print('Storing data...')
		np.savez(DATA_CACHE_VALIDATION, X=X, y=y)
		print('Done')
		print()

	return X, y


if __name__ == '__main__':
	print("Total Characters: ", n_chars)
	print("Total Vocab: ", n_vocab)
	print("Characters:", chars)