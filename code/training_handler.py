import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch_model as models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import math
import numpy as np
import time
import util
from earlyStopping import EarlyStopping

class handler:
	def __init__(self, arg):
		self.gpu = arg.gpu
		self.epoch_size = arg.epoch
		self.batch_size = arg.batch
		self.device = self.device_setting( self.gpu )
		self.hist = dict()

	def device_setting( self, gpu=1 ):
		return 'cpu 'if gpu == -1 else 'cuda:{}'.format(gpu)

	def print_hist( self ):
		string = '===> '

		for k in self.hist.keys():
			val = self.hist[k]

			if type(val) == float:
				string += '{}: {:.6f}\t'.format(k, val)	
			else:
				string += '{}: {}\t'.format(k, val)
		print(string)	
			
	def load_data( self, data, data_type=[torch.LongTensor, torch.LongTensor] ):
		x, y = data

		x = util.padding(x)
	
		data = Variable(data_type[0](x))
		target = Variable(data_type[1](y))

		torch_dataset = Data.TensorDataset(data, target)
		
		print( 'data size:\t{}'.format( len(torch_dataset) ) )
			
		data_loader = Data.DataLoader( dataset=torch_dataset, batch_size=self.batch_size )
			
		return data_loader

	def train( self, model_, train_loader, valid_loader, model_name='checkpoint.pt' ):
		es = EarlyStopping(patience=10, verbose=True)

		optimizer = torch.optim.Adam(model_.parameters())
		#optimizer = torch.optim.SGD(model_.parameters(), lr=3e-4, momentum=0.9)	
		model_.to(self.device)
		loss_func = model_.loss_func()

		start = time.time()
		best_model = None
	
		for epoch in range(self.epoch_size):
			start = time.time()
		
			model_.train()
			train_loss, valid_loss = 0.0, 0.0
			train_acc = 0.0
			for i, (x_, y_) in enumerate(train_loader):
				x_ = x_.to(self.device)
				y_ = y_.to(self.device)
				optimizer.zero_grad()
				y_pred = model_(x_, y_, mode='')
				loss = loss_func(y_pred, y_) 
				loss.backward()
				train_loss += loss.item()*len(x_)
				train_acc += self.accuracy( y_pred, y_ ).item()*len(x_)
				optimizer.step()
			
			self.hist['Epoch'] = epoch+1
			self.hist['time'] = time.time()-start
			self.hist['train_loss'] = train_loss/len(train_loader.dataset)		
			self.hist['train_acc'] = train_acc/len(train_loader.dataset)

			torch.save(model_.state_dict(), model_name)
			
			if valid_loader != None:
				valid_true, valid_pred, valid_loss = self.test(model_, valid_loader, model_name)

				es(valid_loss, model_, model_name)
				
			self.print_hist()
			
			if es.early_stop:	
				print('Early stopping')
				break
				
	def test( self, model_, test_loader, model_name='checkpoint.pt' ):
		model_.load_state_dict(torch.load(model_name))
		model_.to(self.device)
		model_.eval()
		test_loss = 0.0
		test_acc = 0.0
		loss_func = model_.loss_func()

		y_pred, y_true = [], []
		with torch.no_grad():
			for i, (x_, y_) in enumerate(test_loader):
				x_ = x_.to(self.device)
				y_ = y_.to(self.device)
				logit = model_(x_, None, mode='test')
				loss = loss_func(logit, y_)
				
				y_pred.extend(logit)
				y_true.extend(y_)
				test_loss += loss.item()*len(x_)
				test_acc += self.accuracy( logit, y_).item()*len(x_)

		self.hist['val_loss'] = test_loss/len(test_loader.dataset)
		self.hist['val_acc'] = test_acc/len(test_loader.dataset)
		
		return y_true , y_pred, test_loss/len(test_loader.dataset)

	def predict( self, model_, dataset, model_name='checkpoint.pt' ):
		model_.to(self.device)
		model_.load_state_dict(torch.load(model_name))
		model_.eval()
		torch_dataset = self.load_data(dataset, mode='predict')
		print('# of predicted data : {}'.format(len(torch_dataset)))

		test_loader = Data.DataLoader( dataset=torch_dataset )
		_, y_pred, avg_loss = self.test( model_, test_loader )

		return y_pred

	def accuracy( self, y_pred, p_true ):
		return (np.array(list(map(np.argmax, y_pred.detach().cpu()))) == np.array(p_true.cpu())).sum()/len(y_pred)
