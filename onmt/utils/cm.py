
import sys
import math
import numpy as np
from scipy.sparse import csr_matrix, load_npz

#from config import load_parameters
#from keras_wrapper.dataset import loadDataset
#from keras.models import load_model

# from utils.conficence_measure import Neural_CM
# a = Neural_CM('./trained_models/EuTrans_esen_ActorCritic_Adam/Critic/epoch_180.h5', './datasets/Dataset_EuTrans_esen.pkl')

# GET CONFIDENCE -> EN VEZ DE WORD_TARGET ESTARÍA BIEN PASAR TODA LA ORACION HASTA LA FECHA. SOLO HAY QUE COJER LA ULTIMA PALABRA
# SI HAGO LO DE ARRIBA PODRIA HACER METODOS GENERALES EL GET_RATIO_CONFIDENCE Y GET_MEAN_CONFIDENCE

# |=================================|
# |                                 |
# |      /$$$$$$$ /$$$$$$/$$$$      |
# |     /$$_____/| $$_  $$_  $$     |
# |    | $$      | $$ \ $$ \ $$     |
# |    | $$      | $$ | $$ | $$     |
# |    |  $$$$$$$| $$ | $$ | $$     |
# |     \_______/|__/ |__/ |__/     |
# |	                                |
# |=================================|
class CM:
	DIC_EXT = 'txt'
	MAT_EXT = 'npz'
	END_P = '</SEG>'
	SOURCE = 0
	TARGET = 1
	"""
	Class with the basic methods that are needed for calculating the confidence measures
	Functions:
		get_ratio_confidence ->
		get_mean_confidence ->
		get_confidence ->
		file_not_found_alert ->
		no_word_alert ->
	"""
	def __init__(self):
		pass

	def log(self, value):
		try:
			return math.log(value)
		except Exception:
			return -math.inf

	def get_ratio_confidence(self, words_source, words_target, threshold):
		"""
		Calculate the ration confidence measure of a sentence
		:param words_source: List of words from the source sentence
		:param words_target: List of words from the target sentence
		:param threshold: Threshold to mark a word as correct
		:return: Confidence measure
		"""
		correct_words = 0.0

		for pos, word_t in enumerate(words_target):
			prob = self.get_confidence(words_source, words_target[:pos+1], pos+1, len(words_target))
			if prob >= threshold:
				correct_words += 1

		len_sentence = len(words_target)
		confidence = correct_words / len_sentence

		return confidence

	def get_mean_confidence(self, words_source, words_target):
		"""
		Calculate the mean confidence measure of a sentence
		:param words_source: List of words from the source sentence
		:param words_target: List of words form the target sentence
		:return: Confidence measure 
		"""

		if len(words_target) == 0 or len(words_target) == 0:
			return 0.0

		confidence = 0.0
		for pos, word_t in enumerate(words_target):
			value = self.log(self.get_confidence(words_source, words_target[:pos+1], pos+1, len(words_target)))
			confidence += value

		len_sentence = len(words_target)
		confidence = confidence/len_sentence
		confidence = math.exp(confidence)

		return confidence

	def get_confidence(self, sentence_source, word_target, target_pos=None, target_len=None):
		pass

	def file_not_found_alert(self, file_path):
		print('{} not found!'.format(file_path))
		sys.exit()

	def no_word_alert(self, word):
		if self.verbose:
			print(f"'{word}' isn't in the dictionary")

# |=============================================================================================|    
# |                                                                                             |
# |      /$$   /$$                                        /$$                                   |
# |     | $$$ | $$                                       | $$                                   |
# |     | $$$$| $$  /$$$$$$  /$$   /$$  /$$$$$$  /$$$$$$ | $$        /$$$$$$$ /$$$$$$/$$$$      |
# |     | $$ $$ $$ /$$__  $$| $$  | $$ /$$__  $$|____  $$| $$       /$$_____/| $$_  $$_  $$     |
# |     | $$  $$$$| $$$$$$$$| $$  | $$| $$  \__/ /$$$$$$$| $$      | $$      | $$ \ $$ \ $$     |
# |     | $$\  $$$| $$_____/| $$  | $$| $$      /$$__  $$| $$      | $$      | $$ | $$ | $$     |
# |     | $$ \  $$|  $$$$$$$|  $$$$$$/| $$     |  $$$$$$$| $$      |  $$$$$$$| $$ | $$ | $$     |
# |     |__/  \__/ \_______/ \______/ |__/      \_______/|__/       \_______/|__/ |__/ |__/     |
# |                                                                                             |
# |=============================================================================================|
class Neural_CM(CM):
	"""
	Specialization of the confidence measures to cases where we require to use an NMT model
	Functions:
		get_confidence ->
	"""
	def __init__(self, model_path, dataset_path, verbose=False):
		CM.__init__(self)

		print('Loading Parameters from {}'.format('config.py'))
		self.params = load_parameters()
		self.dataset = loadDataset(dataset_path)
		self.unk_id = self.dataset.extra_words['<unk>']
		if 'bpe' in self.params['TOKENIZATION_METHOD'].lower():
			if not self.dataset.BPE_built:
				dataset.build_bpe(self.params.get('BPE_CODES_PATH', self.params['DATA_ROOT_PATH'] + '/training_codes.joint'),separator='@@')
		self.model = load_model(model_path + '.h5')

		self.word2index_y = self.dataset.vocabulary[self.params['OUTPUTS_IDS_DATASET'][0]]['words2idx']
		self.word2index_y[CM.END_P] = 0
		self.word2index_x = self.dataset.vocabulary[self.params['INPUTS_IDS_DATASET'][0]]['words2idx']
		self.word2index_x[CM.END_P] = 0

	def get_max_bleu(self, target_len=7, target_pos=1):
		if target_pos==0 or target_len < 4:
			return 0

		def perfect_gram(n_gram, target_pos, target_len):
			up = 1
			down = target_len - (n_gram-1)
			if target_pos>=n_gram:
				up = target_pos - (n_gram-1)
			else:
				down *= math.pow(2, n_gram-target_pos)

			return up/down


		gram1 = perfect_gram(1, target_pos, target_len)
		gram2 = perfect_gram(2, target_pos, target_len)
		gram3 = perfect_gram(3, target_pos, target_len)
		gram4 = perfect_gram(4, target_pos, target_len)

		return math.exp((math.log(gram1)+math.log(gram2)+math.log(gram3)+math.log(gram4))/4)


	def get_confidence(self, sentence_source, word_target, target_pos=None, target_len=None):
		"""
		"""
		source_idx = []
		for word in sentence_source:
			if word != CM.END_P:
				tok_word = self.dataset.tokenize_bpe(word).split()
				for w in tok_word:
					source_idx.append(self.word2index_x.get(w, self.unk_id))
			else:
				source_idx.append(self.word2index_x[word])

		target_idx = []
		for word in word_target:
			if word != CM.END_P:
				tok_word = self.dataset.tokenize_bpe(word).split()
				for w in tok_word:
					target_idx.append(self.word2index_y.get(w, self.unk_id))
			else:
				target_idx.append(self.word2index_y[word])


		probs = self.model.predict([[source_idx], [target_idx]])
		probs = probs[0][-1][0]

		if target_len >=4:
			diff = self.get_max_bleu(target_len, target_pos) - self.get_max_bleu(target_len, target_pos-1)
			probs = (probs*100)/diff
			if probs >= 100:
				probs = 100
		# OUT NUMPY ARRAY 
		return probs

# |==================================================================================================================================|   
# |                                                                                                                                  |
# |      /$$$$$$   /$$                 /$$     /$$             /$$     /$$                     /$$                                   |
# |     /$$__  $$ | $$                | $$    |__/            | $$    |__/                    | $$                                   |
# |    | $$  \__//$$$$$$    /$$$$$$  /$$$$$$   /$$  /$$$$$$$ /$$$$$$   /$$  /$$$$$$$  /$$$$$$ | $$        /$$$$$$$ /$$$$$$/$$$$      |
# |    |  $$$$$$|_  $$_/   |____  $$|_  $$_/  | $$ /$$_____/|_  $$_/  | $$ /$$_____/ |____  $$| $$       /$$_____/| $$_  $$_  $$     |
# |     \____  $$ | $$      /$$$$$$$  | $$    | $$|  $$$$$$   | $$    | $$| $$        /$$$$$$$| $$      | $$      | $$ \ $$ \ $$     |
# |     /$$  \ $$ | $$ /$$ /$$__  $$  | $$ /$$| $$ \____  $$  | $$ /$$| $$| $$       /$$__  $$| $$      | $$      | $$ | $$ | $$     |
# |    |  $$$$$$/ |  $$$$/|  $$$$$$$  |  $$$$/| $$ /$$$$$$$/  |  $$$$/| $$|  $$$$$$$|  $$$$$$$| $$      |  $$$$$$$| $$ | $$ | $$     |
# |     \______/   \___/   \_______/   \___/  |__/|_______/    \___/  |__/ \_______/ \_______/|__/       \_______/|__/ |__/ |__/     |
# |                                                                                                                                  |
# |==================================================================================================================================|   
class Statistical_CM(CM):
	"""
	Specialization of the conficence measures to cases where we use statistical methods and require to load matrixs
	Functions:
		log ->
		load_dictionaries ->
		load_matrix ->
		load_nonzero_lexicon ->
		get_ratio_confidence ->
		get_mean_confidence ->
		get_confidence ->
		combine_probabilities ->
		get_lexicon_probability ->
	"""
	def __init__(self, model_path, verbose=False):
		CM.__init__(self)

		self.verbose = verbose

		self.probability_matrix = self.load_matrix(model_path)
		self.nonzero_matrix = self.load_nonzero_lexicon()
		self.lexicon_smoothing = 0.003

		self.words2index, self.index2words = self.load_dictionaries(model_path)

	def load_dictionaries(self, model_path):
		"""
		Load the dictionaries to translate words to indices and vice versa
		:param model_path: Path to the file
		:return: words2index & index2words dictionaries
		"""
		try:
			with open(f"{model_path}.{CM.DIC_EXT}", 'r', encoding='utf-8') as f:
					lines = f.read().splitlines()

			words2index = [{CM.END_P:0}, {CM.END_P:0}]
			index2words = [{0:CM.END_P}, {0:CM.END_P}]

			idx = 1
			for word in lines[0].split()[1:]:
				index2words[CM.SOURCE][idx] = word
				words2index[CM.SOURCE][word] = idx
				idx += 1

			idx = 1
			for word in lines[1].split()[1:]:
				index2words[CM.TARGET][idx] = word
				words2index[CM.TARGET][word] = idx
				idx += 1

			return words2index, index2words

		except FileNotFoundError:
			file_not_found_alert('{}.{}'.format(model_path, CM.DIC_EXT))

	def load_matrix(self, model_path):
		"""
		Load the matrix of probabilities
		:param model_path: Path to the file
		:return: csr_matrix with the probabilities [Source x Target]
		"""
		try:
			return load_npz(f"{model_path}.{CM.MAT_EXT}")
		except FileNotFoundError:
			file_not_found_alert('{}.{}'.format(model_path, CM.MAT_EXT))

	def load_nonzero_lexicon(self):
		shape = self.probability_matrix.shape

		nonzero = []
		for idx in range(shape[0]):
			nz = self.probability_matrix[idx, :].count_nonzero()
			nonzero.append(shape[1]-nz)

		return nonzero

	def get_confidence(self, sentence_source, word_target, target_pos=None, target_len=None):
		pass

	def combine_probabilities(self, prob1, prob2, alpha=0.5):
		"""
		Combination of two models
		:params prob1: Log probability 1
		:params prob2: Log probability 2
		:params alpha: Value
		:return: Linear combination
		"""
		prob1 *= alpha 
		if math.isnan(prob1):
			prob1 = 0.0

		prob2 *= (1-alpha)
		if math.isnan(prob2):
			prob2 = 0.0

		return prob1 + prob2

	def get_lexicon_probability(self, word_source, word_target):
		"""
		Get the lexion probability
		:param word_source: Word from the source sentence
		:param word_target: Word from the target sentence
		:return: Probability of the target being the translation of source
		"""
		idx_source = 0
		idx_target = 0
		try:
			idx_source = self.words2index[CM.SOURCE][word_source]
			idx_target = self.words2index[CM.TARGET][word_target]
			prob = self.probability_matrix[idx_source, idx_target]
		except Exception:
			prob = 0.0

		if prob == 0.0:
			prob = self.lexicon_smoothing / self.nonzero_matrix[idx_source]
		else:
			prob *= (1 - self.lexicon_smoothing)

		return prob

#====================================================|
#                                                    |
#     /$$ /$$                             /$$        |
#    |__/| $$                           /$$$$        |
#     /$$| $$$$$$$  /$$$$$$/$$$$       |_  $$        |
#    | $$| $$__  $$| $$_  $$_  $$        | $$        |
#    | $$| $$  \ $$| $$ \ $$ \ $$        | $$        |
#    | $$| $$  | $$| $$ | $$ | $$        | $$        |
#    | $$| $$$$$$$/| $$ | $$ | $$       /$$$$$$      |
#    |__/|_______/ |__/ |__/ |__/      |______/      |
#                                                    |
#====================================================|
class IBM1(Statistical_CM):

	def __init__(self, model_path, verbose=False):
		Statistical_CM.__init__(self, model_path, verbose)

	def get_confidence(self, sentence_source, word_target, target_pos=None, target_len=None):
		"""
		Get the confidence measure of a word in a sentence
		:param sentence_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:return: Max translate probability
		"""
		max_prob = 0.0
		word_target = word_target[-1]

		for word_source in sentence_source:
			prob = self.get_lexicon_probability(word_source, word_target)

			if prob > max_prob:
				max_prob = prob

		return max_prob

#=====================================================|
#                                                     |
#     /$$ /$$                            /$$$$$$      |
#    |__/| $$                           /$$__  $$     |
#     /$$| $$$$$$$  /$$$$$$/$$$$       |__/  \ $$     |
#    | $$| $$__  $$| $$_  $$_  $$        /$$$$$$/     |
#    | $$| $$  \ $$| $$ \ $$ \ $$       /$$____/      |
#    | $$| $$  | $$| $$ | $$ | $$      | $$           |
#    | $$| $$$$$$$/| $$ | $$ | $$      | $$$$$$$$     |
#    |__/|_______/ |__/ |__/ |__/      |________/     |
#                                                     |
#=====================================================|
class IBM2(Statistical_CM):

	def __init__(self, model_path, alignment_path, alpha=0.5, verbose=False):
		Statistical_CM.__init__(self, model_path, verbose)
		self.alignment_matrix = self.load_alignment_model(alignment_path)
		self.alpha = alpha

	def load_alignment_model(self, alignment_path):
		"""
		Load the alignment probabilities
		:param alignment_path: Path to the file
		:return: alignment_matrix
		"""
		try:
			alignment_matrix = dict()
			with open(alignment_path, 'r', encoding='utf-8') as f:
				for line in f:
					elements = line.split()
					s_pos = int(elements[0])
					t_pos = int(elements[1])
					s_length = int(elements[2])
					t_length = int(elements[3])
					probability = float(elements[4])

					if s_length not in alignment_matrix:
						alignment_matrix[s_length] = dict()
					if t_length not in alignment_matrix[s_length]:
						new_table = np.zeros((s_length+1, t_length+1))
						alignment_matrix[s_length][t_length] = new_table

					current_table = alignment_matrix[s_length][t_length]
					current_table[s_pos][t_pos] = probability

			return alignment_matrix	    
		except FileNotFoundError:
			file_not_found_alert(alignment_path)

	def get_alignment_probability(self, s_pos, t_pos, s_len, t_len):
		"""
		Get the probability of the current alignment
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:return: Probability of the alignment
		"""
		if s_len in self.alignment_matrix and t_len in self.alignment_matrix[s_len]:
			return self.alignment_matrix[s_len][t_len][s_pos][t_pos]
		else:
			return 0.0

	def get_confidence(self, sentence_source, word_target, target_pos, target_len):
		"""
		Get the confidence measure of a word in a sentence
		:param sentence_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:param target_pos: Position of the target word
		:param target_len: Length of the target sentence
		:return: Max translate probability
		"""
		source_len = len(sentence_source)
		word_target = word_target[-1]

		lex_prob = self.log(self.get_lexicon_probability(CM.END_P, word_target))
		ali_prob = self.log(self.get_alignment_probability(0, target_pos, source_len, target_len))
		max_prob = self.combine_probabilities(lex_prob, ali_prob, self.alpha)

		for pos, word in enumerate(sentence_source):
			lex_prob = self.log(self.get_lexicon_probability(word, word_target))
			ali_prob = self.log(self.get_alignment_probability(pos+1, target_pos, source_len, target_len))
			prob =  self.combine_probabilities(lex_prob, ali_prob, self.alpha)

			if prob > max_prob:
				max_prob = prob

		return math.exp(max_prob)

#=========================================================================================|
#                                                                                         |
#     /$$$$$$$$                   /$$            /$$$$$$  /$$ /$$                         |
#    | $$_____/                  | $$           /$$__  $$| $$|__/                         |
#    | $$    /$$$$$$   /$$$$$$$ /$$$$$$        | $$  \ $$| $$ /$$  /$$$$$$  /$$$$$$$      |
#    | $$$$$|____  $$ /$$_____/|_  $$_/        | $$$$$$$$| $$| $$ /$$__  $$| $$__  $$     |
#    | $$__/ /$$$$$$$|  $$$$$$   | $$          | $$__  $$| $$| $$| $$  \ $$| $$  \ $$     |
#    | $$   /$$__  $$ \____  $$  | $$ /$$      | $$  | $$| $$| $$| $$  | $$| $$  | $$     |
#    | $$  |  $$$$$$$ /$$$$$$$/  |  $$$$/      | $$  | $$| $$| $$|  $$$$$$$| $$  | $$     |
#    |__/   \_______/|_______/    \___/        |__/  |__/|__/|__/ \____  $$|__/  |__/     |
#                                                                 /$$  \ $$               |
#                                                                |  $$$$$$/               |
#                                                                 \______/                |
#                                                                                         |
#=========================================================================================|
class Fast_Align(Statistical_CM):

	def __init__(self, model_path, alpha=0.5, prob_0 = 0, tension = 4, verbose=False):
		Statistical_CM.__init__(self, model_path, verbose)

		self.alpha = alpha
		self.prob_0 = prob_0
		self.tension = tension

	def get_alignment_probability(self, s_pos, t_pos, s_len, t_len, norm_factor):
		"""
		Get the probability of the current alignment
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:param norm_factor: Normalization factor to use the fast_align equation
		:return: Probability of the alignment
		"""
		if s_pos == 0:
			return self.prob_0
		elif s_pos <= s_len:
			foo = self.get_e(s_pos, t_pos, s_len, t_len)
			foo /= norm_factor
			foo *= (1 - self.prob_0)
			return foo
		else:
			return 0.0

	def get_h(self, s_pos, t_pos, s_len, t_len):
		"""
		Implementation of the equation: h(i,j,m,n) = -|frac{i}{m}-frac{j}{n}|"
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:return: -|frac{t_pos}{t_len}-frac{s_pos}{s_len}|
		"""
		a = t_pos/t_len
		b = s_pos/s_len

		a -= b
		a = -abs(a)

		return a

	def get_e(self, s_pos, t_pos, s_len, t_len):
		"""
		Implementation of the equation: e^{lambda h(i,j,m,n)}
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:return: e^{self.tension h(t_pos,s_pos,t_len,s_len)}
		"""
		a = self.get_h(s_pos, t_pos, s_len, t_len)
		a *= self.tension
		a = math.exp(a)
		return a

	def get_s(self, s_pos, t_pos, s_len, t_len, r, l):
		"""
		Implementation of the equation: s_l(g,r) = g frac{1-r^l}{1-r}
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:param r:
		:param l:
		:return: e(t_pos, s_pos, t_len, s_len) frac{1-r^l}{1-r}
		"""
		g = self.get_e(s_pos, t_pos, s_len, t_len)

		a = 1 - math.pow(r, l)
		b = 1 - r
		a /= b

		return g*a

	def get_normalize_factor(self, t_pos, s_len, t_len):
		"""
		Calculate the normalization factor for the current target position
		Equation: s_{j1}(e^{lambda h(i,j1,m,n)}, r) + s_{n-j2}(e^{lambda h(i,j2,m,n)}, r)
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:return: Normalization factor
		"""
		j_up = math.floor((t_pos/t_len)*s_len)
		j_dw = j_up + 1

		r = math.exp(-self.tension/s_len)

		s_up = self.get_s(j_up, t_pos, s_len, t_len, r, j_up)
		s_dw = self.get_s(j_dw, t_pos, s_len, t_len, r, s_len-j_dw+1)

		return s_up + s_dw

	def get_confidence(self, sentence_source, word_target, target_pos, target_len):
		"""
		Get the confidence measure of a word in a sentence
		:param sentence_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:param target_pos: Position of the target word
		:param target_len: Length of the target sentence
		:return: Max Translate Probability
		"""
		word_target = word_target[-1]
		source_len = len(sentence_source)
		norm_factor = self.get_normalize_factor(target_pos, source_len, target_len)

		lex_prob = self.log(self.get_lexicon_probability(CM.END_P, word_target))
		ali_prob = self.log(self.get_alignment_probability(0, target_pos, source_len, target_len, norm_factor))
		max_prob = self.combine_probabilities(lex_prob, ali_prob, self.alpha)

		for pos, word in enumerate(sentence_source):
			lex_prob = self.log(self.get_lexicon_probability(word, word_target))
			ali_prob = self.log(self.get_alignment_probability(pos+1, target_pos, source_len, target_len, norm_factor))
			prob =  self.combine_probabilities(lex_prob, ali_prob, self.alpha)
			if prob > max_prob:
				max_prob = prob

		return math.exp(max_prob)

#=================================================|
#                                                 |
#      /$$                                        |
#     | $$                                        |
#     | $$$$$$$  /$$$$$$/$$$$  /$$$$$$/$$$$       |
#     | $$__  $$| $$_  $$_  $$| $$_  $$_  $$      |
#     | $$  \ $$| $$ \ $$ \ $$| $$ \ $$ \ $$      |
#     | $$  | $$| $$ | $$ | $$| $$ | $$ | $$      |
#     | $$  | $$| $$ | $$ | $$| $$ | $$ | $$      |
#     |__/  |__/|__/ |__/ |__/|__/ |__/ |__/      |
#                                                 |
#=================================================|
class HMM(Statistical_CM):
	def __init__(self, model_path, align_model_path, alpha=0.5, verbose=False):
		Statistical_CM.__init__(self, model_path, verbose)
		self.alignment_matrix = self.load_alignment_model(align_model_path)

		self.alpha = alpha

		self.current_source = None
		self.dynamic_matrix = []

	def load_alignment_model(self, alignment_path):
		"""
		Load the alignment model from the *.hhmm.* file
		:param alignment_path: Path to the file with the alignments
		:return: Dictionary with the probabilities
		"""
		try:
			with open(alignment_path, 'r', encoding='utf-8') as f:
				alignment_matrix = dict()
				previous = 0

				for line in f:
					line = line.rstrip('\n')
					if line != "":
						action, data = line.split(':', 1)
						if action == 'Distribution for':
							previous, data = data.split('  ', 1)
							previous = int(previous.split(':', -1)[1])
							alignment_matrix[previous] = dict()

							data = data.replace(' ', '');
							for pair in data.split(';'):
								if pair != '':
									current, prob = pair.split(':')
									current = int(current)
									prob = float(prob)
									alignment_matrix[previous][current] = prob

						elif action == 'SUM':
							value = float(data)
							for key in alignment_matrix[previous]:
								alignment_matrix[previous][key] /= value

						elif action == 'FULL-SUM':
							value = float(data)

				return alignment_matrix

		except FileNotFoundError:
			file_not_found_alert(alignment_path)

	def get_alignment_probability(self, current_pos, previous_pos):
		"""
		Obtain the alignment probability from the previous and current position of the source sentence
		:param current_pos: Position of the current source word
		:param previous_pos: Position of the previous source word
		:return: p(j|j',J)
		"""
		return self.alignment_matrix[previous_pos][previous_pos - current_pos]

	def generate_dynamic_matrix(self, sentence_source, target_pos):
		"""
		Creates and extends the structure to calculate dynamicaly which previous source position was the better
		:param sentence_source: Sentence from the source corpus
		:param target_pos: Position of the current target word
		"""
		if self.current_source != sentence_source:
			self.current_source = sentence_source
			self.dynamic_matrix = []

		while len(self.dynamic_matrix) < target_pos:
			self.dynamic_matrix.append([-math.inf for i in range(len(sentence_source))])

	def get_confidence(self, sentence_source, word_target, pos_target, len_target=None):
		"""
		Calculates the confidence of the current target word
		:param sentence_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:param pos_target: Position of the target word in the target sentence
		:param target_len: Length of the target sentence
		:return: Confidence
		"""
		word_target = word_target[-1]
		s_source = sentence_source[:]
		s_source.insert(0, CM.END_P)
		self.generate_dynamic_matrix(s_source, pos_target)

		pos_target -= 1
		probabilities = []

		if pos_target == 0:
			for pos_source, word_source in enumerate(s_source): 
				prob_lex = self.get_lexicon_probability(word_source, word_target)
				self.dynamic_matrix[pos_target][pos_source] = self.log(prob_lex)
				probabilities.append(prob_lex)
		else:
			for pos_source, word_source in enumerate(s_source):
				max_prob = -math.inf
				max_prev = 0
				for previous in range(len(s_source)):
					prob = self.log(self.get_alignment_probability(pos_source, previous))
					prob += self.dynamic_matrix[pos_target-1][previous]
					if prob >= max_prob:
						max_prob = prob
						max_prev = previous

				prob_lex = self.log(self.get_lexicon_probability(word_source, word_target))
				prob_ali = self.log(self.get_alignment_probability(pos_source, max_prev))

				self.dynamic_matrix[pos_target][pos_source] = max_prob + prob_lex
				probabilities.append(self.combine_probabilities(prob_lex, prob_ali, self.alpha))

		max_prob = -math.inf
		for prob in probabilities:
			if prob > max_prob:
				max_prob = prob

		return math.exp(max_prob)
