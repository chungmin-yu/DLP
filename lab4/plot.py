#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

class plotConfusionMatrix:
	def __init__(self, model_name, pretrained_flag):
		self.model_name = model_name
		self.pretrained_flag = pretrained_flag
	def plot(self):
		predictions=[]
		labels=[]

		if self.pretrained_flag == True:
			with open("record/" + self.model_name + "_"  + "pretrained_prediction.txt", "r") as f:
				for line in f.readlines():
					s = line.strip('\n')
					predictions.append(float(s))
			with open("record/" + self.model_name + "_"  + "pretrained_label.txt", "r") as f:
				for line in f.readlines():
					s = line.strip('\n')
					labels.append(float(s))
		else:
			with open("record/" + self.model_name + "_"  + "prediction.txt", "r") as f:
				for line in f.readlines():
					s = line.strip('\n')
					predictions.append(float(s))
			with open("record/" + self.model_name + "_"  + "label.txt", "r") as f:
				for line in f.readlines():
					s = line.strip('\n')
					labels.append(float(s))

		print(len(predictions))
		print(len(labels))
		ConfusionMatrixDisplay.from_predictions(labels, predictions, normalize = 'true', cmap=plt.cm.Blues)
		print(accuracy_score(labels, predictions))
		plt.show()

class plotResult:
	def __init__(self, model_name, epoch_num):
		self.model_name = model_name
		self.epoch_num = epoch_num
	def plot(self):
		epoch=[i for i in range(self.epoch_num)]
		pretrained_train=[]
		pretrained_test=[]
		train=[]
		test=[]

		with open("record/" + self.model_name + "_"  + "pretrained_accuracy_train.txt", "r") as f:
			for line in f.readlines():
				s = line.strip('\n')
				pretrained_train.append(float(s))
		with open("record/" + self.model_name + "_"  + "pretrained_accuracy_test.txt", "r") as f:
			for line in f.readlines():
				s = line.strip('\n')
				pretrained_test.append(float(s))
		with open("record/" + self.model_name + "_"  + "accuracy_train.txt", "r") as f:
			for line in f.readlines():
				s = line.strip('\n')
				train.append(float(s))
		with open("record/" + self.model_name + "_"  + "accuracy_test.txt", "r") as f:
			for line in f.readlines():
				s = line.strip('\n')
				test.append(float(s))

		title_name = 'Result comparaison (' + self.model_name + ')'
		plt.title(title_name, fontsize=18)
		#plt.plot(epoch, acc, 'C0o-', linewidth=1, markersize=2, label="xxx")
		plt.plot(epoch, pretrained_train, '-', linewidth=2, label="training with pretraining")
		plt.plot(epoch, pretrained_test, '-', linewidth=2, label="testing with pretraining")
		plt.plot(epoch, train, '-', linewidth=2, label="training without pretraining")
		plt.plot(epoch, test, '-', linewidth=2, label="testing without pretraining")
		plt.xlabel('Epoch', fontsize=12) 
		plt.ylabel('Accuracy(%)', fontsize=12) 
		plt.legend(loc = "upper left", fontsize=9)
		plt.show()