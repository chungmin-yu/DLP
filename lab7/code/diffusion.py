import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image


class Diffusion:
	def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"): 
		self.noise_steps = noise_steps
		self.beta_start = beta_start
		self.beta_end = beta_end

		self.beta = self.prepare_noise_schedule().to(device)
		self.alpha = 1. - self.beta
		self.alpha_hat = torch.cumprod(self.alpha, dim=0)

		self.img_size = img_size
		self.device = device

	# linear beta schedule
	# def prepare_noise_schedule(self):
	# 	return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

	# cosine beta schedule
	# def prepare_noise_schedule(self, s=0.008):
	# 	steps = self.noise_steps + 1
	# 	x = torch.linspace(0, self.noise_steps, steps)
	# 	alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
	# 	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	# 	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	# 	return torch.clip(betas, 0.0001, 0.9999)

	# quadratic beta schedule
	# def prepare_noise_schedule(self):
	# 	return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.noise_steps) ** 2

	# sigmoid betas chedule
	def prepare_noise_schedule(self):
		betas = torch.linspace(-6, 6, self.noise_steps)
		return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start

	def noise_images(self, x, t):
		sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
		sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
		a = torch.randn_like(x)
		return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * a, a

	def sample_timesteps(self, n):
		return torch.randint(low=1, high=self.noise_steps, size=(n,))

	def sample(self, model, n, labels, cfg_scale=3):
		model.eval()
		with torch.no_grad():
			x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
			for i in reversed(range(1, self.noise_steps)):
				t = (torch.ones(n) * i).long().to(self.device)
				predicted_noise = model(x, t, labels)
				if cfg_scale > 0:
					uncond_predicted_noise = model(x, t, None)
					predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
				alpha = self.alpha[t][:, None, None, None]
				alpha_hat = self.alpha_hat[t][:, None, None, None]
				beta = self.beta[t][:, None, None, None]
				if i > 1:
					noise = torch.randn_like(x)
				else:
					noise = torch.zeros_like(x)
				x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

		model.train()
		return x