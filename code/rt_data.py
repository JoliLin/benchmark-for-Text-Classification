import LM
import nltk
import numpy as np
import sys
import util
from collections import deque

class rt_data:
	def __init__( self, path, test_size=0.2, splitted=True, split_token=' ' ):
		self.path = path
		self.test_size = test_size

		data = self.preprocess(self.path, split_token)
		self.dataset = self.train_test_split( data ) if splitted else data

	def load_data( self ):
		return self.dataset

	def preprocess( self, path, split_token=' ' ):
		data = open(path, encoding='utf-8', errors='ignore').readlines()
		np.random.seed(0)
		np.random.shuffle(data)

		y = deque()
		x = deque()

		for i in data:
			y_, x_ = i.split( split_token, 1 )
			
			x_nltk = util.normalizeString(x_)
			x_nltk = nltk.word_tokenize(x_nltk)
			x_nltk = util.removeStop(x_nltk)
			x_nltk = ' '.join(x_nltk)
		
			x.append(x_nltk)
			y.append(int(y_))
		
		return LM.ONE_hot(list(x)).load_LM(), list(y)
		#return LM.BOW(list(x)).load_LM(), list(y)

	def train_test_split( self, data, test_size=0.2 ):
		x, y = data
		test_len = int(0.2*len(x))

		train_x = x[test_len:]
		train_y = y[test_len:]
		test_x = x[:test_len]
		test_y = y[:test_len]

		return (train_x, train_y), (test_x, test_y)
