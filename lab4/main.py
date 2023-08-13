#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy
import argparse
import os
import torchvision.models as models
from dataloader import RetinopathyLoader
from plot import plotConfusionMatrix
from plot import plotResult

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='resnet50')
parser.add_argument('-p', '--pretrained', type=int, default=1)
parser.add_argument('-s', '--save', type=str, nargs='+')
parser.add_argument('-o', '--others', type=str, nargs='+')
args = parser.parse_args()

model_name = args.model
pretrained_flag = args.pretrained
batch_size = 16 if model_name == 'resnet50' else 64
epoch_num = 25

if args.others != None and "draw" in args.others:
	# plot result
	plotObject = plotConfusionMatrix(model_name, pretrained_flag)
	plotObject.plot()
	# plotObject2 = plotResult(model_name, epoch_num)
	# plotObject2.plot()

else: 
	# load data from custom dataloader
	train_data = RetinopathyLoader(root = 'data/', mode = 'train')
	test_data = RetinopathyLoader(root = 'data/', mode = 'test')

	# load data
	train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers=4)
	test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False, num_workers=4)

	if args.others != None and "time" in args.others:
		# record cuda time of training & testing
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()

	if model_name == 'resnet18':
		if pretrained_flag:
			model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
			for param in model.parameters():
				param.requires_grad = False
		else:
			model = models.resnet18(weights=None)
			for param in model.parameters():
				param.requires_grad = True
		ftrs_num = model.fc.in_features
		model.fc = nn.Linear(ftrs_num, 5)
	elif model_name == 'resnet50': 
		if pretrained_flag:
			model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
			for param in model.parameters():
				param.requires_grad = False
		else:
			model = models.resnet50(weights=None)
			for param in model.parameters():
				param.requires_grad = True
		ftrs_num = model.fc.in_features
		model.fc = nn.Linear(ftrs_num, 5)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using device: ' + str(device))
	model.to(device)

	# for pretrained: feature extraction (only update fc.parameters, other parameters set false)
	optimizer_feature = torch.optim.SGD(model.fc.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5e-4)
	optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5e-4)
	criterion = nn.CrossEntropyLoss()
	### weighted loss
	# nSamples = [20656, 1955, 4210, 698, 581]
	# baseline = nSamples[0]
	# normedWeights = [baseline / x for x in nSamples]
	# print("each class weights: " + str(normedWeights))
	# normedWeights = torch.FloatTensor(normedWeights).to(device)
	# criterion = nn.CrossEntropyLoss(weight = normedWeights)

	max_accuracy = -1
	max_epoch = -1
	max_prediction = []
	max_label = []
	if args.others != None and "test" in args.others:
		if pretrained_flag:
			model.load_state_dict(torch.load("weight/pretrained_" + model_name  + ".pt"))
		else: 
			model.load_state_dict(torch.load("weight/" + model_name  + ".pt"))

		total_test = 0
		correct_test = 0
		prediction_list=[]
		label_list=[]
		model.eval()
		with torch.no_grad():
			for i, (data, label) in enumerate(test_loader):
				data = data.to(device, dtype = torch.float)
				label = label.to(device, dtype = torch.long)

				if i % 50 == 0:
					print("finish ", i , " batch...")

				# no need to calculate gradient and loss function

				# forward propagation
				output = model(data) 

				# get predictions from the maximum value
				prediction = torch.max(output.data, 1)[1]

				# total number of labels
				total_test += len(label)

				# total correct predictions
				correct_test += (prediction == label).float().sum()

				# record the prediction & label
				for p in prediction.cpu().tolist():
					prediction_list.append(p)
				for l in label.cpu().tolist():
					label_list.append(l)

		# calculate accuracy
		accuracy = 100 * (correct_test / total_test)
		print("testing accuracy: ", accuracy)

	else:
		accuracy_train = []
		accuracy_test = []
		feature_extraction = pretrained_flag
		for epoch in range(epoch_num):
			# training process
			total_loss = 0
			total_train = 0
			correct_train = 0
			model.train()

			# first 5 epoch perform feature extraction when using pretrained model
			if pretrained_flag and epoch < 5:
				print("\nfeature_extracting...")
			else:
				feature_extraction = False
				print("\nfine tuning...")
				for param in model.parameters():
					param.requires_grad = True

			for i, (data, label) in enumerate(train_loader):
				data = data.to(device, dtype = torch.float)
				label = label.to(device, dtype = torch.long)

				if i % 250 == 0:
					print(data.shape)

				# clear gradient
				if feature_extraction:
					optimizer_feature.zero_grad()
				else:
					optimizer.zero_grad()

				# forward propagation
				output = model(data) 

				# calculate cross entropy (loss function)
				loss = criterion(output, label) 
				total_loss += float(loss)

				# get predictions from the maximum value
				prediction = torch.max(output.data, 1)[1]

				# total number of labels
				total_train += len(label)

				# total correct predictions
				correct_train += (prediction == label).float().sum()

				# Calculate gradients
				loss.backward()

				# Update parameters
				if feature_extraction:
					optimizer_feature.step()
				else:
					optimizer.step()

			# calculate accuracy
			accuracy = 100 * (correct_train / total_train)
			accuracy_train.append(accuracy.item())

			print("\nepoch ", epoch + 1, ":")
			print("trainig accuracy: ", accuracy, "  loss: ", total_loss)

			# testing process
			total_test = 0
			correct_test = 0
			prediction_list=[]
			label_list=[]
			model.eval()
			with torch.no_grad():			
				for i, (data, label) in enumerate(test_loader):
					data = data.to(device, dtype = torch.float)
					label = label.to(device, dtype = torch.long)

					# no need to calculate gradient and loss function

					# forward propagation
					output = model(data) 

					# get predictions from the maximum value
					prediction = torch.max(output.data, 1)[1]

					# total number of labels
					total_test += len(label)

					# total correct predictions
					correct_test += (prediction == label).float().sum()

					# record the prediction & label
					for p in prediction.cpu().tolist():
						prediction_list.append(p)
					for l in label.cpu().tolist():
						label_list.append(l)

			# calculate accuracy
			accuracy = 100 * (correct_test / total_test)
			accuracy_test.append(accuracy.item())

			print("testing accuracy: ", accuracy)

			# record performance info for the best epoch
			if accuracy.item() > max_accuracy:
				max_accuracy = accuracy.item()
				max_epoch = epoch + 1
				max_prediction = prediction_list
				max_label = label_list
				if args.save != None and "weight" in args.save:
					path = "weight"
					# Check whether the specified path exists or not
					isExist = os.path.exists(path)
					if not isExist:
						os.makedirs(path)
					# save model
					print("epoch " + str(max_epoch) + " save weights")
					if pretrained_flag:
						torch.save(model.state_dict(), "weight/pretrained_" + model_name  + ".pt")
					else:
						torch.save(model.state_dict(), "weight/" + model_name  + ".pt")

			if args.save != None and "record" in args.save:
				path = "record"
				# Check whether the specified path exists or not
				isExist = os.path.exists(path)
				if not isExist:
					os.makedirs(path)
				# save record
				if pretrained_flag:
					with open("record/" + model_name + "_"  + "pretrained_prediction.txt", "w") as f:
						for i in max_prediction:
							f.write(str(i) + "\n")
					with open("record/" + model_name + "_"  + "pretrained_label.txt", "w") as f:
						for i in max_label:
							f.write(str(i) + "\n")
					with open("record/" + model_name + "_"  + "pretrained_accuracy_train.txt", "w") as f:
						for i in accuracy_train:
							f.write(str(i) + "\n")
					with open("record/" + model_name + "_"  + "pretrained_accuracy_test.txt", "w") as f:
						for i in accuracy_test:
							f.write(str(i) + "\n")
				else:
					with open("record/" + model_name + "_"  + "prediction.txt", "w") as f:
						for i in max_prediction:
							f.write(str(i) + "\n")
					with open("record/" + model_name + "_"  + "label.txt", "w") as f:
						for i in max_label:
							f.write(str(i) + "\n")
					with open("record/" + model_name + "_"  + "accuracy_train.txt", "w") as f:
						for i in accuracy_train:
							f.write(str(i) + "\n")
					with open("record/" + model_name + "_"  + "accuracy_test.txt", "w") as f:
						for i in accuracy_test:
							f.write(str(i) + "\n")

		print("\n" + model_name + " testing has max accuracy " + str(max_accuracy) + "% at epoch " + str(max_epoch))
		
		
		if args.others != None and "time" in args.others:
			# print execution time
			end.record()
			torch.cuda.synchronize()
			print("execution time: " + str(start.elapsed_time(end)/1000) + "s")



