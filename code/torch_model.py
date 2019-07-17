import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self, Kernal=10, classes=2, embed_size=64):
		super(CNN, self).__init__()
		self.loss = nn.CrossEntropyLoss()
		
		self.embed = nn.Embedding(100000, embed_size)
		self.conv3 = nn.Conv2d(1, Kernal, (3, embed_size), padding=(1, 0))
		self.conv4 = nn.Conv2d(1, Kernal, (4, embed_size), padding=(2, 0))
		self.conv5 = nn.Conv2d(1, Kernal, (5, embed_size), padding=(2, 0))
		self.final = nn.Linear(3*Kernal, classes)

	def loss_func(self):
		return self.loss

	def main_task(self, x):
		x = self.embed(x).unsqueeze(1)

		x3 = F.relu(self.conv3(x)).squeeze(3)
		x4 = F.relu(self.conv4(x)).squeeze(3)
		x5 = F.relu(self.conv5(x)).squeeze(3)

		x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
		x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)
		x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2)

		x = torch.cat((x3, x4, x5), 1)

		rating = self.final(x)
		return rating

	def forward(self, x_ , y_, mode='train'):
		y_rating = self.main_task(x_)
		if mode == 'train':
			return self.loss(y_rating, y_)

		else:
			return y_rating

class MultiHeadAtt(nn.Module):
	def __init__(self, n_head=2, embed_size=16, d_head=2, classes=2, max_len=256 ):
		super(MultiHeadAtt, self).__init__()
		self.loss = nn.CrossEntropyLoss()
		self.embed = nn.Embedding(100000, embed_size)
		self.post = nn.Embedding(300, embed_size)
		
		self.n_head = n_head
		self.d_model = embed_size
		self.d_head = d_head
		self.max_len = max_len

		self.q_net = nn.Linear(embed_size, n_head*d_head, bias=False)
		self.kv_net = nn.Linear(embed_size, 2*n_head*d_head, bias=False)

		self.o_net = nn.Linear(n_head*d_head, embed_size, bias=False)

		self.scale = 1/(d_head ** 0.5)
		self.drop = nn.Dropout(p=0.2)

		self.final = nn.Linear(embed_size*max_len, classes)

	def loss_func(self):
		return self.loss

	def main_task(self, h):
		head_q = self.q_net(h)
		head_k, head_v = torch.chunk(self.kv_net(h), 2, -1)
		
		head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
		head_k = head_k.view(h.size(0), h.size(1), self.n_head, self.d_head)
		head_v = head_v.view(h.size(0), h.size(1), self.n_head, self.d_head)

		att_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
		att_score.mul_(self.scale)

		att_prob = F.softmax(att_score, dim=1)

		att_vec = torch.einsum('ijbn, jbnd->ibnd', (att_prob, head_v))
		att_vec = att_vec.contiguous().view(att_vec.size(0), att_vec.size(1), self.n_head*self.d_head)

		att_out = self.o_net(att_vec)
		output = h+att_out
	
		return output

	def forward(self, x_, y_, mode='train'):
		x_ = self.embed(x_)
		output = self.main_task(x_)

		y_rating = self.final(output.view(-1, self.d_model*self.max_len))
	
		if mode == 'train':		
			return self.loss(y_rating, y_)
		else:
			return y_rating

class MLP(nn.Module):
	def __init__(self, dim=64, classes=2):
		super(MLP, self).__init__()
		self.loss = nn.CrossEntropyLoss()
		
		self.embed = nn.Embedding(100000, dim)
		self.in_net = nn.Linear(dim, 16)
		self.o_net = nn.Linear(16, classes)
		
	def loss_func(self):
		return self.loss

	def main_task(self, h):
		h = self.embed(h)

		h = F.avg_pool2d(h, (h.size(1), 1)).squeeze(1)
		
		h = self.in_net(h)
		h = F.relu(h)
	
		h = self.o_net(h)
		return h
	
	def forward(self, x_, y_, mode='train'):
		y_rating = self.main_task(x_)
	
		if mode =='train':
			return self.loss(y_rating, y_)
		else:
			return y_rating
