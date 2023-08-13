#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# fix random seed to check accuracy
#np.random.seed(4)

# lab spec hint
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
	return np.multiply(x, 1.0 - x)

def ReLU(x):
	x = np.maximum(0.0, x)
	return x

def derivative_ReLU(x):
	x[x <= 0] = 0
	x[x > 0] = 1
	return x

def tanh(x):
	return np.tanh(x)

def derivative_tanh(x):
	return 1 - np.square(x)

# You need to use the following generate functions to create your inputs x, y.
def generate_linear(n = 100):
	pts = np.random.uniform(0, 1, (n, 2))
	inputs = []
	labels = []
	for pt in pts:
		inputs.append([pt[0], pt[1]])
		distance = (pt[0] - pt[1]) / 1.414
		if pt[0] > pt[1]:
			labels.append(0)
		else:
			labels.append(1)
	return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
	inputs = []
	labels = []

	for i in range(11):
		inputs.append([0.1*i, 0.1*i])
		labels.append(0)

		if 0.1*i == 0.5:
			continue

		inputs.append([0.1*i, 1-0.1*i])
		labels.append(1)
	
	return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
	plt.subplot(1, 2, 1)
	plt.title('Ground truth', fontsize=18)
	for i in range(x.shape[0]):
		if y[i] == 0:
			plt.plot(x[i][0], x[i][1], 'ro')
		else:
			plt.plot(x[i][0], x[i][1], 'bo')

	plt.subplot(1, 2, 2)
	plt.title('Predict result', fontsize=18)
	for i in range(x.shape[0]):
		if pred_y[i] == 0:
			plt.plot(x[i][0], x[i][1], 'ro')
		else:
			plt.plot(x[i][0], x[i][1], 'bo')

	plt.show()

def show_learning_curve(epoch, loss):
	plt.title('Learning Curve', fontsize=18)
	#plt.plot(epoch, loss, 'bo-', label='xxx')
	plt.plot(epoch, loss, 'bo-', linewidth=1, markersize=2)
	plt.xlabel('epoch', fontsize=12) 
	plt.ylabel('loss', fontsize=12) 
	#plt.legend(loc = "best", fontsize=10)
	plt.show()

# TODO
# Implement simple neural networks with two hidden layers.
def activation(func, arg):
	if func == "sigmoid":
		return sigmoid(arg)
	elif func == "ReLU":
		return ReLU(arg)
	elif func == "tanh":
		return tanh(arg)
	else:
		return arg

def derivative_activation(func, arg):
	if func == "sigmoid":
		return derivative_sigmoid(arg)
	elif func == "ReLU":
		return derivative_ReLU(arg)
	elif func == "tanh":
		return derivative_tanh(arg)
	else:
		return 1

def forwarding_pass(func, x, w1, w2, w3):
	a0 = x
	z1 = w1 @ a0
	a1 = activation(func, z1)
	z2 = w2 @ a1
	a2 = activation(func, z2)
	z3 = w3 @ a2
	a3 = sigmoid(z3)
	pred_y = a3
	return pred_y, a0, a1, a2, a3

# given pred_y, y, a3, w3, a2, w2, a1, w1
# we want to know dJ_dw3, dJ_dw2, dJ_dw1
def backpropagation(func, pred_y, y, a3, w3, a2, w2, a1, w1, a0):
	eps = 0.0001
	dJ_da3 = -( y / (pred_y + eps) - (1 - y) / (1 - pred_y + eps) ) # derivative of cross-entropy
	dJ_dz3 = derivative_sigmoid(a3) * dJ_da3
	dJ_dw3 = dJ_dz3 @ a2.T

	dJ_da2 = w3.T @ dJ_dz3
	dJ_dz2 = derivative_activation(func, a2) * dJ_da2
	dJ_dw2 = dJ_dz2 @ a1.T

	dJ_da1 = w2.T @ dJ_dz2
	dJ_dz1 = derivative_activation(func, a1) * dJ_da1
	dJ_dw1 = dJ_dz1 @ a0.T

	return dJ_dw1, dJ_dw2, dJ_dw3

def weight_update(optimizer, learning_rate, n, dJ_dw1, dJ_dw2, dJ_dw3, w1, w2, w3, m1, m2, m3):
	if optimizer == True:
		m1 = 0.9 * m1 - learning_rate * (dJ_dw1 / n)
		m2 = 0.9 * m2 - learning_rate * (dJ_dw2 / n)
		m3 = 0.9 * m3 - learning_rate * (dJ_dw3 / n)
		w1 = w1 + m1
		w2 = w2 + m2
		w3 = w3 + m3
	else:
		w1 = w1 - learning_rate * (dJ_dw1 / n)
		w2 = w2 - learning_rate * (dJ_dw2 / n)
		w3 = w3 - learning_rate * (dJ_dw3 / n)
	return w1, w2, w3, m1, m2, m3

def loss(pred_y, y):
	# MSE
	# np.mean((pred_y - y) ** 2)
	# cross-entropy
	# -1/n * (y * ln(pred_y) + (1-y) * ln(1-pred_y))
	eps = 0.0001
	loss = -(1 / y.shape[1]) * (y @ np.log(pred_y + eps).T + (1 - y) @ np.log(1 - pred_y + eps).T)
	return float(loss)

if __name__=='__main__':
	x, y = generate_linear()
	#x, y = generate_XOR_easy()
	x = x.T
	y = y.T

	neuron = 10 # weight size
	epoch = 5000
	learning_rate = 0.1
	activation_func = "sigmoid"
	epoch_list=[]
	loss_list=[]	

	w1 = np.random.randn(neuron, 2)
	w2 = np.random.randn(neuron, neuron)
	w3 = np.random.randn(1, neuron)
	
	# Momentum Optimizer
	optimizer = False 
	m1 = np.random.randn(neuron, 2)
	m2 = np.random.randn(neuron, neuron)
	m3 = np.random.randn(1, neuron)

	# training
	print("Training ...")
	for i in range(epoch):
		pred_y, a0, a1, a2, a3 = forwarding_pass(activation_func, x, w1, w2, w3)
		L = loss(pred_y, y)
		loss_list.append(L)
		epoch_list.append(i)
		dJ_dw1, dJ_dw2, dJ_dw3 = backpropagation(activation_func, pred_y, y, a3, w3, a2, w2, a1, w1, a0)
		w1, w2, w3, m1, m2, m3 = weight_update(optimizer, learning_rate, neuron, dJ_dw1, dJ_dw2, dJ_dw3, w1, w2, w3, m1, m2, m3)
		# print intermediate result
		if (i+1) % 100 == 0:
			print("epoch", i+1, end=' ')
			print("loss :", L)
			#accuracy = (1 - np.sum( np.abs( y - np.round(pred_y) ) ) / y.shape[1]) * 100
			#print("accuracy :", accuracy, "%")
	
	#show_learning_curve(epoch_list, loss_list)

	# testing
	print("\nTesting ...")
	x_test, y_test = generate_linear()
	#x_test, y_test = generate_XOR_easy()
	x_test = x_test.T
	y_test = y_test.T
	pred_y, a0, a1, a2, a3 = forwarding_pass(activation_func, x_test, w1, w2, w3)
	accuracy = (1 - np.sum( np.abs( y_test - np.round(pred_y) ) ) / y_test.shape[1]) * 100
	print(pred_y)
	print("accuracy :", accuracy, "%")
	show_result(x_test.T, y_test.T, np.round(pred_y).T)

