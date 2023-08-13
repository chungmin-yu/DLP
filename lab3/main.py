#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy
import argparse
import os
from dataloader import read_bci_data
from EEG import EEGNet
from DeepConv import DeepConvNet
from ShallowConv import ShallowConvNet
from plot import plotResult

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='EEGNet')
parser.add_argument('-s', '--save', type=str, nargs='+')
parser.add_argument('-o', '--others', type=str, nargs='+')
args = parser.parse_args()

model_name = args.model
epoch_num = 500

if args.others != None and "draw" in args.others:
	# plot result
	plotObject = plotResult(model_name, epoch_num)
	plotObject.plot()
else:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#device = torch.device('cpu')
	print('Using device: ' + str(device))

	# load data & convert numpy to tensor
	train_data, train_label, test_data, test_label = read_bci_data()
	train_data = torch.from_numpy(train_data)
	train_label = torch.from_numpy(train_label)
	test_data = torch.from_numpy(test_data)
	test_label = torch.from_numpy(test_label)

	# package tensor, like zip function
	train_tensor = torch.utils.data.TensorDataset(train_data, train_label)
	test_tensor = torch.utils.data.TensorDataset(test_data, test_label)

	# loading data
	train_loader = torch.utils.data.DataLoader(train_tensor, batch_size = 256, shuffle = True)
	test_loader = torch.utils.data.DataLoader(test_tensor, batch_size = 256, shuffle = True)

	activation_func = [nn.ReLU(), nn.LeakyReLU(), nn.ELU()]
	func = ["ReLU", "LeakyReLU", "ELU"]
	acc=[]

	for func, name in zip(activation_func, func):
		if name == "ReLU":
			print("\nUsing ReLU ...")
		elif name == "LeakyReLU":
			print("\nUsing LeakyReLU ...")
		elif name == "ELU":
			print("\nUsing ELU ...")

		if args.others != None and "time" in args.others:
			# record cuda time of training & testing
			start = torch.cuda.Event(enable_timing=True)
			end = torch.cuda.Event(enable_timing=True)
			start.record()

		# Network configuration
		if model_name=="EEGNet":
			model = EEGNet(func, device)
		elif model_name=="DeepConvNet":
			model = DeepConvNet(func, device)
		elif model_name=="ShallowConvNet":
			model = ShallowConvNet(func, device)

		# setting device, optimizer, loss function, number of epochs
		model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.01)
		criterion = nn.CrossEntropyLoss()

		if args.others != None and "test" in args.others:
			model.load_state_dict(torch.load("weight/" + model_name + "_"  + name + ".pt"))

			# testing process
			total_test = 0
			correct_test = 0
			model.eval()
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

			# calculate accuracy
			max_accuracy = 100 * (correct_test / total_test)
			acc.append(max_accuracy.item())

		else:
			accuracy_train = []
			accuracy_test = []
			for epoch in range(epoch_num):
				# training process
				total_loss = 0
				total_train = 0
				correct_train = 0
				model.train()
				for i, (data, label) in enumerate(train_loader):
					data = data.to(device, dtype = torch.float)
					label = label.to(device, dtype = torch.long)

					# clear gradient
					optimizer.zero_grad()

					# forward propagation
					output = model(data) 

					# calculate cross entropy (loss function)
					loss = criterion(output, label) 
					total_loss += loss

					# get predictions from the maximum value
					prediction = torch.max(output.data, 1)[1]

					# total number of labels
					total_train += len(label)

				    # total correct predictions
					correct_train += (prediction == label).float().sum()

					# Calculate gradients
					loss.backward()

					# Update parameters
					optimizer.step()

				# calculate accuracy
				accuracy = 100 * (correct_train / total_train)
				accuracy_train.append(accuracy.item())

				if epoch % 10 == 9:
					print("\nepoch ", epoch + 1, ":")
					print("trainig accuracy: ", accuracy, "  loss: ", total_loss)

				# testing process
				total_test = 0
				correct_test = 0
				model.eval()
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

				# calculate accuracy
				accuracy = 100 * (correct_test / total_test)
				accuracy_test.append(accuracy.item())

				if epoch % 10 == 9:
					print("testing accuracy: ", accuracy)

			max_accuracy = max(accuracy_test)
			print("\n" + name + " has max accuracy " + str(max_accuracy) + "% at epoch " + str(accuracy_test.index(max_accuracy)))
			acc.append(max_accuracy)
			
			if args.others != None and "time" in args.others:
				# print execution time
				end.record()
				torch.cuda.synchronize()
				print("execution time: " + str(start.elapsed_time(end)/1000) + "s")

			if args.save != None and "record" in args.save:
				path = "record"
				# Check whether the specified path exists or not
				isExist = os.path.exists(path)
				if not isExist:
					os.makedirs(path)
				with open("record/" + model_name + "_"  + name + "_train.txt", 'w') as f:
					for i in accuracy_train:
						f.write(str(i) + '\n')
				with open("record/" + model_name + "_"  + name + "_test.txt", 'w') as f:
					for i in accuracy_test:
						f.write(str(i) + '\n')
			if args.save != None and "weight" in args.save:
				path = "weight"
				# Check whether the specified path exists or not
				isExist = os.path.exists(path)
				if not isExist:
					os.makedirs(path)
				# save model
				torch.save(model.state_dict(), "weight/" + model_name + "_"  + name + ".pt")


	print("")	
	print("ReLU has max accuracy ", acc[0], "%")
	print("LeakyReLU has max accuracy ", acc[1], "%")
	print("ELU has max accuracy ", acc[2], "%")	


