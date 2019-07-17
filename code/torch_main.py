import args
import rt_data as rt
import sys
import torch
import torch_model as models
import training_handler
import util

from tensorflow import keras
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

thandler = training_handler.handler(args.process_command())	

def RUN_SVC( data ):
	print('SVC')
	(train_data, train_labels), (test_data, test_labels) = data

	clf = SVC(C=0.1, gamma='auto')
	clf.fit( util.padding(train_data), train_labels )
	y_pred = clf.predict( util.padding(test_data) )
	print('Accuracy: {}'.format(accuracy_score(test_labels, y_pred)))
	
def data_loader( data_, data_type=[torch.LongTensor, torch.LongTensor] ):
	(train_data, train_labels), (test_data, test_labels) = data_
	
	train_size = int(len(train_data)*0.1)

	valid_data = train_data[:train_size]	
	valid_labels = train_labels[:train_size]
	train_loader = thandler.load_data((train_data[train_size:], train_labels[train_size:]), data_type)
	valid_loader = thandler.load_data((valid_data, valid_labels), data_type)
	test_loader = thandler.load_data((test_data, test_labels), data_type)

	return train_loader, valid_loader, test_loader

def RUN( model_, data_, model_name ):
	train_loader, valid_loader, test_loader = data_

	print(model_)
	total = sum(p.numel() for p in model_.parameters() if p.requires_grad)
	print('# of para: {}'.format(total))	
	
	thandler.train( model_, train_loader, valid_loader, model_name )
	y_true, y_pred, avg_loss = thandler.test( model_, test_loader, model_name )
	
	score = util.hit_rate([i.detach().cpu() for i in y_true], [i.detach().cpu() for i in y_pred])
	print('Accuracy: {}'.format(score))

def main_process( data, classes):
	#RUN_SVC(data)

	data = data_loader(data)
	
	RUN( models.MLP(16, classes), data, 'MLP.pt' )
	RUN( models.CNN(classes=classes, embed_size=64), data, 'CNN.pt' )
	RUN( models.MultiHeadAtt(2, 16, 2, classes=classes), data, 'attention.pt' )

