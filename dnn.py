# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 23:35:16 2018

@author: andy
"""

import numpy as np

def ReLu(v):
	return np.maximum(v,0)

def grad_ReLu(v):
	return (v > 0) + np.zeros(v.shape)
	
def loss(yhat, ytrue):
	ydiff = ytrue-yhat
	return np.dot(ydiff,ydiff)/np.shape(ytrue)[0]

def grad_loss(yhat, ytrue):
	ydiff = ytrue-yhat
	return -2*ydiff/np.shape(ytrue)[0]
	
class dnn:
	def __init__(self, num_layers, dim, learning_rate = 0.001):
		self.W_lst = []
		self.b_lst = []
		self.num_layers = num_layers
		self.d = dim
		self.lambdaval= learning_rate
		
	def fit(self, x, y, 
			num_iter = 1000000,
			tol=1e-6, 
			beta1 = 0.9,
			beta2 = 0.999,
			eps = 1e-8):
		N = x.shape[0]
		a_lst = [np.copy(x)]
		z_lst = [np.copy(x)]
		for i in range(0, self.num_layers):
			if (i ==0):
				self.W_lst.append(np.random.randn(self.d,N))
				self.b_lst.append(np.random.randn(self.d))
				a_lst.append(np.zeros(self.d))
				z_lst.append(np.zeros(self.d))
			elif (i==self.num_layers-1):
				self.W_lst.append(np.random.randn(N, self.d))
				self.b_lst.append(np.random.randn(N))
				a_lst.append(np.zeros(N))
				z_lst.append(np.zeros(N))

			else:
				self.W_lst.append(np.random.randn(self.d, self.d))
				self.b_lst.append(np.random.randn(self.d))
				a_lst.append(np.zeros(self.d))
				z_lst.append(np.zeros(self.d))
		#self.W_lst= [W1,W2,W3]
		#self.b_lst =[ b1,b2,b3]
		previous_loss = np.inf
		for _it in range(0, num_iter+1):
			current_loss = loss(a_lst[-1], y)
			if abs(previous_loss - current_loss) < tol:
				break
			previous_loss = current_loss
			if ((_it) % 100 == 0):
				print(_it, current_loss)
			# evaluate each node (forward)
			for alpha in range(1, self.num_layers+1):
				z = np.matmul(self.W_lst[alpha-1], a_lst[alpha-1]) + self.b_lst[alpha-1]
				z_lst[alpha] = z
				a_lst[alpha] = ReLu(z)
				
			# backward induction
			# params for Adam
			delta = grad_loss(a_lst[-1], y)
			
			w_mom_dict={}
			b_mom_dict={}
			w_reg_dict={}
			b_reg_dict={}

			for alpha in np.arange(self.num_layers-1, 0, -1):
				W = self.W_lst[alpha]
				a = a_lst[alpha]
				dW = np.outer(delta, a)
				db = delta
						
				if alpha not in w_mom_dict:
					w_mom = dW
					b_mom = db
					w_reg = dW**2
					b_reg = db**2
				else:
					w_mom = beta1*w_mom_dict[alpha] + (1-beta1)*dW
					b_mom = beta1*b_mom_dict[alpha] + (1-beta1)*db
					w_reg = beta2*w_reg_dict[alpha] + (1-beta2)*(dW**2)
					b_reg = beta2*b_reg_dict[alpha] + (1-beta2)*(db**2)				
				w_mom_dict[alpha] = w_mom
				b_mom_dict[alpha] = b_mom
				w_reg_dict[alpha] = w_reg
				b_reg_dict[alpha] = b_reg
				
				w_mom_tilde = w_mom / (1-beta1**(_it+1))
				b_mom_tilde = b_mom / (1-beta1**(_it+1))
				w_reg_tilde = w_reg / (1-beta2**(_it+1))
				b_reg_tilde = b_reg / (1-beta2**(_it+1))
				
				self.W_lst[alpha] = W - self.lambdaval * (w_mom_tilde/(w_reg_tilde**0.5 + eps))
				self.b_lst[alpha] = self.b_lst[alpha] - self.lambdaval * (b_mom_tilde/(b_reg_tilde**0.5 + eps))
								
				delta = np.multiply(np.matmul(self.W_lst[alpha].T, delta),grad_ReLu(z_lst[alpha]))
		return self.W_lst, self.b_lst
	
	def predict(self, x):
		v = x
		for alpha in range(1, self.num_layers+1):
			v = np.matmul(self.W_lst[alpha-1], v) + self.b_lst[alpha-1]
			v = ReLu(v)		
		return v
if __name__ == "__main__":
	N = 10
	d = 5
	np.random.seed(2)
	x = np.random.randn(N)
	W1 = np.random.randn(d,N)
	W2 = np.random.randn(d,d)
	W3 = np.random.randn(N,d)
	b1 = np.random.randn(d)
	b2 = np.random.randn(d)
	b3 = np.random.randn(N)
	y = ReLu(np.matmul(W1, x)+b1)
	y = ReLu(np.matmul(W2, y) + b2)
	y = ReLu(np.matmul(W3, y)+ b3)
	y = y + 0.1*np.random.randn(N)
	clf = dnn(3,d)
	clf.fit(x,y)
	yhat = clf.predict(x)
	ydiff = y - yhat
	theloss = np.mean(ydiff**2)
	print("loss:",theloss)