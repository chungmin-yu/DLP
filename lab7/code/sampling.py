import os
import torch
import argparse
from PIL import Image
import torchvision.utils as vutils
from torchvision import transforms
from diffusion import *
from dataset import CLEVRDataset, get_test_conditions, get_new_test_conditions
from models import UNet_conditional
from evaluator import evaluation_model

parser = argparse.ArgumentParser()

parser.add_argument('--size',type=int,default=32,help='image size')
parser.add_argument('--iter',type=int,default=10)
parser.add_argument('--out_dir',type=str,default='./demo/log')
parser.add_argument('--num_classes',type=int,default=24,help='number of class')
parser.add_argument('--checkpoint1',type=str,default='./demo/best_model1.pt')
parser.add_argument('--checkpoint2',type=str,default='./demo/best_model2.pt')

args = parser.parse_args()

def save_images(images, path, **kwargs):
	images = (images.clamp(-1, 1) + 1) / 2
	images = (images * 255).type(torch.uint8)
	grid = vutils.make_grid(images, **kwargs)
	ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
	im = Image.fromarray(ndarr)
	im.save(path)

def main():
	#initial setting
	os.makedirs(args.out_dir,exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
	print('Using device: ' + str(device))

	# load model
	model = UNet_conditional(num_classes=args.num_classes).to(device)
	diffusion = Diffusion(img_size=args.size, device=device)
	eval_model = evaluation_model()
	model.load_state_dict(torch.load(args.checkpoint1))

	# test
	print("\ntest.json")
	best_score1 = 0
	li1=[]
	test_conditions = get_test_conditions().to(device)
	for i in range(1, args.iter+1):
		torch.manual_seed(i)
		print('Torch seed: ' + str(i))
		sampled_images1 = diffusion.sample(model, n=len(test_conditions), labels=test_conditions)
		save_images(sampled_images1, os.path.join(args.out_dir, f"demo_1_{i}.png"),nrow=8)
		sampled_images1 = transforms.Resize([64,64])(sampled_images1)
		score1 = eval_model.eval(sampled_images1.float(), test_conditions)
		print('score1: ' + str(score1))
		li1.append(score1)
	with open('./{}/generation_record1.txt'.format(args.out_dir), 'w') as f:
		for i in range(len(li1)):
			f.write('seed: ' + str(i+1) + ',score: ' + str(li1[i]) + '\n')
	print('max score: ' + str(max(li1)))
	print('avg score: ' + str(sum(li1) / len(li1)))
	print()


	model.load_state_dict(torch.load(args.checkpoint2))

	# new_test
	print("new_test.json")
	best_score2 = 0
	li2=[]
	new_test_conditions = get_new_test_conditions().to(device)
	for i in range(1, args.iter+1):
		torch.manual_seed(i)
		print('Torch seed: ' + str(i))
		sampled_images2 = diffusion.sample(model, n=len(new_test_conditions), labels=new_test_conditions)
		save_images(sampled_images2, os.path.join(args.out_dir, f"demo_2_{i}.png"),nrow=8)
		sampled_images2 = transforms.Resize([64,64])(sampled_images2)
		score2 = eval_model.eval(sampled_images2.float(), new_test_conditions)
		print('score2: ' + str(score2))
		li2.append(score2)
	with open('./{}/generation_record2.txt'.format(args.out_dir), 'w') as f:
		for i in range(len(li2)):
			f.write('seed: ' + str(i+1) + ',score: ' + str(li2[i]) + '\n')
	print('max score: ' + str(max(li2)))
	print('avg score: ' + str(sum(li2) / len(li2)))

if __name__ == '__main__':
	main()