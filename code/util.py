import numpy as np
import re
import sys
import time
from collections import deque
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error as mse

stop = set(stopwords.words('english'))

def sent2words( sent ):
	return word_tokenize(sent) 

def normalizeString(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
	string = re.sub(r" : ", ":", string)
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " ( ", string) 
	string = re.sub(r"\)", " ) ", string) 
	string = re.sub(r"\?", " ? ", string) 
	string = re.sub(r"\s{2,}", " ", string)   
	return string.strip().lower()

def removeStop(lst):
	return [word for word in lst if word not in stop]

def key2value( target, dic ):
	return dic[target]

def list2file( file_name, data ):
	with open( file_name, 'w') as f:
		for i in range(len(data)):
			f.write('{}\n'.format(data[i]))

def padding( x, max_len=256 ):
	return [i+(max_len-len(i))*[0] if len(i) < max_len else i[:max_len] for i in x]

def rmse(y_true, y_pred):
	return mse(y_true, y_pred) ** 0.5

def hit_rate(y_true, y_pred):
	hit = 0
	for i in zip(y_true, y_pred):
		if i[0] == np.argmax(np.array(i[1])):
			hit += 1

	return float(hit)/len(y_true)
