#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy

class ShallowConvNet(nn.Module):
	def __init__(self, activation_func, device):
		super(ShallowConvNet, self).__init__()

		self.device = device
		self.conv0 = nn.Conv2d(1, 40, kernel_size = (1, 13))
		self.conv1 = nn.Sequential(
			nn.Conv2d(40, 40, kernel_size = (2, 1), bias = False),
			nn.BatchNorm2d(40, eps = 1e-5, momentum = 0.1),
			activation_func
		)
		self.pool = nn.Sequential(
			nn.AvgPool2d(kernel_size = (1, 35), stride = (1, 7)),
			nn.Dropout(p = 0.5)
		)

		self.classify = nn.Linear(4040, 2)

	def forward(self, X):
		out = self.conv0(X.to(self.device))
		out = self.conv1(out)
		#out = torch.square(out) # activation function
		out = self.pool(out)
		#out = torch.log(torch.clip(out, 1e-7, 10000))
		out = out.view(out.shape[0], -1)  #flatten/resize
		out = self.classify(out)
		#out = nn.functional.softmax(out, dim = 0)
		return out


