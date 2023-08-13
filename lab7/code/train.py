import os
import argparse
import torch
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.utils as vutils
from torchvision import transforms
from dataset import CLEVRDataset, get_test_conditions, get_new_test_conditions
from diffusion import *
from models import UNet_conditional, EMA
from evaluator import evaluation_model

parser = argparse.ArgumentParser()

parser.add_argument('-bs',type=int,default=48,help='batch size')
parser.add_argument('-ep',type=int,default=300,help='epoch')
parser.add_argument('-lr',type=float,default=3e-4,help='learning rate')
parser.add_argument('-dp','--dropout',type=float,default=0.2,help='dropout')
parser.add_argument('--size',type=int,default=32,help='image size')
parser.add_argument('--in_channels',type=int,default=3,help='input channels')
parser.add_argument('--num_classes',type=int,default=24,help='number of class')
parser.add_argument('--save_dir',type=str,default='./checkpoint')
parser.add_argument('--num_workers',type=int,default=4,help='num_workers')

args = parser.parse_args()

def save_images(images, path, **kwargs):
	images = (images.clamp(-1, 1) + 1) / 2
	images = (images * 255).type(torch.uint8)
	grid = vutils.make_grid(images, **kwargs)
	ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
	im = Image.fromarray(ndarr)
	im.save(path)

def train():
	ckpt_loc = os.path.join(args.save_dir,f'{datetime.today().strftime("%m-%d-%H-%M-%S")}_DDPM')
	mod_loc = os.path.join(ckpt_loc,'model')
	img_loc = os.path.join(ckpt_loc,'generate')
	os.makedirs(ckpt_loc,exist_ok=True)
	os.makedirs(mod_loc,exist_ok=True)
	os.makedirs(img_loc,exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
	print('Using device: ' + str(device))

	# Lists to keep track of progress
	losses = []
	best_score1 = 0
	best_score2 = 0
	test_conditions = get_test_conditions().to(device)
	new_test_conditions = get_new_test_conditions().to(device)

	# model
	model = UNet_conditional(num_classes=args.num_classes).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	criterion = nn.MSELoss()
	diffusion = Diffusion(img_size=args.size, device=device)
	ema = EMA(0.995)
	ema_model = copy.deepcopy(model).eval().requires_grad_(False)
	
	dataset_train = CLEVRDataset()
	train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
	eval_model = evaluation_model()
	
	print("Starting Training Loop...")
	train_bar = tqdm(total=args.ep)
	for epoch in range(1,args.ep+1):		
		for i, (images, conditions) in enumerate(train_loader):
			total_loss = 0
			images = images.to(device)
			labels = conditions.to(device)
			t = diffusion.sample_timesteps(images.shape[0]).to(device)
			x_t, noise = diffusion.noise_images(images, t)
			if np.random.random() < 0.1:
				labels = None
			predicted_noise = model(x_t, t, labels)
			loss = criterion(noise, predicted_noise)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			ema.step_ema(ema_model, model)

			total_loss += loss.item()
			
		train_bar.update(1)

		# Save Losses for plotting later
		losses.append(total_loss)
		print('epoch ' + str(epoch) + ' loss: ' + str(total_loss))
		with open(os.path.join(ckpt_loc, 'loss_record.txt'), 'a') as f:
			f.write(str(total_loss) + "\n")

		# evaluate test dataset
		if epoch % 5 == 0:
			# test
			#sampled_images1 = diffusion.sample(model, n=len(test_conditions), labels=test_conditions)
			ema_sampled_images1 = diffusion.sample(ema_model, n=len(test_conditions), labels=test_conditions)
			#save_images(sampled_images1, os.path.join(img_loc, f"sample_1_{epoch}.png"),nrow=8)
			save_images(ema_sampled_images1, os.path.join(img_loc, f"ema_1_{epoch}.png"),nrow=8)
			# new_test
			#sampled_images2 = diffusion.sample(model, n=len(new_test_conditions), labels=new_test_conditions)
			ema_sampled_images2 = diffusion.sample(ema_model, n=len(new_test_conditions), labels=new_test_conditions)
			#save_images(sampled_images2, os.path.join(img_loc, f"sample_2_{epoch}.png"),nrow=8)
			save_images(ema_sampled_images2, os.path.join(img_loc, f"ema_2_{epoch}.png"),nrow=8)
			# save & evaluate
			torch.save(model.state_dict(), os.path.join(mod_loc, f"{epoch:0>4}ckpt.pt"))
			torch.save(ema_model.state_dict(), os.path.join(mod_loc, f"{epoch:0>4}ckpt.pt"))
			ema_sampled_images1 = transforms.Resize([64,64])(ema_sampled_images1)           
			score1 = eval_model.eval(ema_sampled_images1.float(), test_conditions)
			ema_sampled_images2 = transforms.Resize([64,64])(ema_sampled_images2)
			score2 = eval_model.eval(ema_sampled_images2.float(), new_test_conditions)
			
			if score1 > best_score1:
				best_score1 = score1
				torch.save(ema_model.state_dict(),  f"best_model1.pt")	

			if score2 > best_score2:
				best_score2 = score2
				torch.save(ema_model.state_dict(), f"best_model2.pt")
			
			print('epoch ' + str(epoch) + ' score1: ' + str(score1) + ' score2: ' + str(score2) + '\n' )
			with open(os.path.join(ckpt_loc, 'score1_record.txt'), 'a') as f:
				f.write(str(score1) + "\n")	
			with open(os.path.join(ckpt_loc, 'score2_record.txt'), 'a') as f:
				f.write(str(score2) + "\n")		

	print('\nbest score1: ' + str(best_score1))
	print('\nbest score2: ' + str(best_score2))

if __name__ == '__main__':
	train()