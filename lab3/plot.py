#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class plotResult:
	def __init__(self, model_name, epoch_num):
		self.model_name = model_name
		self.epoch_num = epoch_num
	def plot(self):
		color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
		epoch=[i for i in range(self.epoch_num)]
		ReLU_train=[]
		ReLU_test=[]
		LeakyReLU_train=[]
		LeakyReLU_test=[]
		ELU_train=[]
		ELU_test=[]

		with open("record/" + self.model_name + "_ReLU_train.txt", 'r') as f:
			for line in f.readlines():
				s = line.strip('\n')
				ReLU_train.append(float(s))
		with open("record/" + self.model_name + "_ReLU_test.txt", 'r') as f:
			for line in f.readlines():
				s = line.strip('\n')
				ReLU_test.append(float(s))
		with open("record/" + self.model_name + "_LeakyReLU_train.txt", 'r') as f:
			for line in f.readlines():
				s = line.strip('\n')
				LeakyReLU_train.append(float(s))
		with open("record/" + self.model_name + "_LeakyReLU_test.txt", 'r') as f:
			for line in f.readlines():
				s = line.strip('\n')
				LeakyReLU_test.append(float(s))
		with open("record/" + self.model_name + "_ELU_train.txt", 'r') as f:
			for line in f.readlines():
				s = line.strip('\n')
				ELU_train.append(float(s))
		with open("record/" + self.model_name + "_ELU_test.txt", 'r') as f:
			for line in f.readlines():
				s = line.strip('\n')
				ELU_test.append(float(s))

		title_name = 'Activation function comparaison (' + self.model_name + ')'
		plt.title(title_name, fontsize=18)
		#plt.plot(epoch, acc, 'C0o-', linewidth=1, markersize=2, label="xxx")
		plt.plot(epoch, ReLU_train, '-', linewidth=1, label="ReLU_train")
		plt.plot(epoch, ReLU_test, '-', linewidth=1, label="ReLU_test")
		plt.plot(epoch, LeakyReLU_train, '-', linewidth=1, label="LeakyReLU_train")
		plt.plot(epoch, LeakyReLU_test, '-', linewidth=1, label="LeakyReLU_test")
		plt.plot(epoch, ELU_train, '-', linewidth=1, label="ELU_train")
		plt.plot(epoch, ELU_test, '-', linewidth=1, label="ELU_test")
		plt.xlabel('Epoch', fontsize=12) 
		plt.ylabel('Accuracy(%)', fontsize=12) 
		plt.legend(loc = "lower right", fontsize=10)
		plt.show()
