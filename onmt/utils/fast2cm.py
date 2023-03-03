from scipy.sparse import csr_matrix, save_npz
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.DEBUG,
			format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(2)

END_P = '</SEG>'
SOURCE = 0
TARGET = 1


def load_vocab(file_name):
	index2words = [{0:END_P}, {0:END_P}]
	words2index = [{END_P:0}, {END_P:0}]

	with open(file_name, 'r') as f:
		lines = f.read().splitlines()

	src_idx = 1
	trg_idx = 1
	for line in lines:
		src_word, trg_word, prob = line.split()

		if src_word not in words2index[SOURCE]:
			index2words[SOURCE][src_idx] = src_word
			words2index[SOURCE][src_word] = src_idx
			src_idx += 1
		if trg_word not in words2index[TARGET]:
			index2words[TARGET][trg_idx] = trg_word
			words2index[TARGET][trg_word] = trg_idx
			trg_idx += 1

	return index2words, words2index

def load_model(file_name, words2index):
	source_len = len(words2index[SOURCE])
	target_len = len(words2index[TARGET])

	logger.debug("Len source: {}".format(source_len))
	logger.debug("Len target: {}".format(target_len))
	logger.debug("Size Matrix: {}".format(source_len*target_len))

	probabilities = np.zeros((source_len, target_len))
	with open(file_name, 'r') as f:
		lines = f.read().splitlines()

	for line in lines:
		src_word, trg_word, prob = line.split()
		src_idx = words2index[SOURCE][src_word]
		trg_idx = words2index[TARGET][trg_word]
		prob = float(prob)

		probabilities[src_idx][trg_idx] = np.exp(prob)

	probabilities = csr_matrix(probabilities)
	return probabilities

def save_model(file_name, probabilities):
	save_npz(file_name, probabilities)

def save_vocab(file_name, index2words):
	with open(file_name, 'w') as f:
		for lang in (SOURCE, TARGET):
			for w in index2words[lang].values():
				f.write(w+' ')
			f.write('\n')

def get_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument("-m", "--model", type=str, help="Model Name")
	parser.add_argument("-out", "--output", type=str, help="Output of the model")

	return parser.parse_args()

def main():
	args = get_arguments()

	index2words, words2index = load_vocab(args.model) 
	model = load_model(args.model, words2index)

	save_vocab("{}.txt".format(args.output), index2words)
	save_model("{}.npz".format(args.output), model)

if __name__ == '__main__':
	main()
