from collections import Counter, deque
from itertools import chain

class BOW:
	def __init__( self, corpus ):
		self.corpus = corpus
		self.LM = self.preprocess( self.corpus )
		self.key_id = None
		self.dim = 0

	def load_LM( self ):
		return self.LM

	def to_array( self, dic ):
		embed = [0] * self.dim

		for k in dic.keys():
			embed[self.key_id.get(k)] = dic[k]
		
		return embed

	def preprocess( self, corpus, mode='TF' ):
		tf = [Counter(c.split(' ')) for c in corpus]

		keys = chain.from_iterable([list(i.keys()) for i in tf])

		self.key_id = {i[1]:i[0] for i in enumerate(set(keys))}
		self.dim = len(self.key_id.keys())
		
		if mode == 'TF':
			embed_corpus = list(map(self.to_array, tf))
		'''
		elif mode == 'TF-IDF':
			idf = Counter(keys)

			embed = [0] * len(self.key_id)
			
		elif mode == 'binary':
			embed = [0] * len(self.key_id)
		'''	
		return embed_corpus

class ONE_hot:
	def __init__( self, corpus ):
		self.corpus = corpus
		self.LM = self.preprocess( self.corpus )

	def load_LM( self ):
		return self.LM

	def preprocess( self, corpus ):
		c_ = [c.split(' ') for c in corpus]
		key_id = list(set(chain.from_iterable(c_)))

		return [list(map(key_id.index, k)) for k in c_]

