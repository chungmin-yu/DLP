import os
import torch
import random
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.lstm import gaussian_lstm, lstm
from dataset import bair_robot_pushing_dataset
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, plot_rec, plot_demo, finn_eval_seq, pred

torch.backends.cudnn.benchmark = True

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
	parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
	parser.add_argument('--batch_size', default=12, type=int, help='batch size')
	parser.add_argument('--log_dir', default='./best_model', help='base directory to save logs')
	parser.add_argument('--model_dir', default='', help='base directory to save logs')
	parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
	parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
	parser.add_argument('--niter', type=int, default=160, help='number of epochs to train for')
	parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
	parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
	parser.add_argument('--tfr_start_decay_epoch', type=int, default=60, help='The epoch that teacher forcing ratio become decreasing')
	parser.add_argument('--tfr_decay_step', type=float, default=0.01, help='The decay step size of teacher forcing ratio (0 ~ 1)')
	parser.add_argument('--tfr_lower_bound', type=float, default=0.0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
	parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
	parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
	parser.add_argument('--kl_anneal_cycle', type=int, default=4, help='The number of cycle for kl annealing (if use cyclical mode)')
	parser.add_argument('--seed', default=1, type=int, help='manual seed')
	parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
	parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
	parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
	parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
	parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
	parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
	parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
	parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
	parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
	parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
	parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
	parser.add_argument('--cuda', default=True, action='store_true') 

	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	if args.cuda:
		assert torch.cuda.is_available(), 'CUDA is not available.'
		device = 'cuda'
	else:
		device = 'cpu'
	
	assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
	assert 0 <= args.tfr and args.tfr <= 1
	assert 0 <= args.tfr_start_decay_epoch 
	assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

	
	# load model and continue training from checkpoint
	saved_model = torch.load('best_model/model.pth')
	start_epoch = saved_model['last_epoch']
	print(f'Using pre-trained: best_model/model.pth')

	print("Random Seed: ", args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	print(args)
	# ------------ build the models  --------------

	frame_predictor = saved_model['frame_predictor']
	posterior = saved_model['posterior']
	decoder = saved_model['decoder']
	encoder = saved_model['encoder']
	
	
	# --------- transfer to device ------------------------------------
	frame_predictor.to(device)
	posterior.to(device)
	encoder.to(device)
	decoder.to(device)

	# --------- load a dataset ------------------------------------
	test_data = bair_robot_pushing_dataset(args, 'test')

	test_loader = DataLoader(test_data,
							num_workers=args.num_workers,
							batch_size=args.batch_size,
							shuffle=False,
							drop_last=True,
							pin_memory=True)

	test_iterator = iter(test_loader)

	modules = {
		'frame_predictor': frame_predictor,
		'posterior': posterior,
		'encoder': encoder,
		'decoder': decoder,
	}

	frame_predictor.eval()
	encoder.eval()
	decoder.eval()
	posterior.eval()

	psnr_list = []
	progress = tqdm(total=len(test_data)//args.batch_size)
	for data_idx in range(len(test_data)//args.batch_size):
		try:
			test_seq, test_cond = next(test_iterator)
		except StopIteration:
			test_iterator = iter(test_loader)
			test_seq, test_cond = next(test_iterator)

		test_seq = test_seq.permute(1, 0, 2, 3, 4).to(device) # change seq to [batch size, frame num, 3, 64, 64]
		test_cond = test_cond.permute(1, 0, 2).to(device) # change condition to [batch size, frame num, 7]

		pred_seq = pred(test_seq, test_cond, modules, args)
		_, _, psnr = finn_eval_seq(test_seq[args.n_past:args.n_past+args.n_future], pred_seq[args.n_past:])

		with open('./{}/generation_record.txt'.format(args.log_dir), 'a') as f:
			f.write('idx ' + str(data_idx) + ': '+ str(np.mean(np.concatenate(psnr))))

		psnr_list.append(np.mean(np.concatenate(psnr)))
		progress.update(1)

	ave_psnr = np.mean(np.array(psnr_list))
	print(' avg: ' + str(ave_psnr))
	plot_rec(test_seq, test_cond, modules, 0, args)
	plot_demo(test_seq, test_cond, modules, args)
		
if __name__ == '__main__':
	main()