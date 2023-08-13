import pandas as pd
import numpy as np
from cProfile import label
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import argparse

def plot_psnr(log_dir):
	n_iter = 160
	n_iter=[i for i in range(n_iter)]
	psnr = []	

	with open('./{}/psnr_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			psnr.append(float(s))

	plt.title('Training psnr curve', fontsize=18)
	plt.plot(n_iter, psnr, 'o-', linewidth=2, markersize=3, label="psnr")
	plt.xlabel('iters', fontsize=12) 
	plt.ylabel('score', fontsize=12) 
	plt.legend(loc = "lower right", fontsize=12)
	plt.grid(axis='y')
	plt.show()

def plot_score(log_dir):
	n_iter = 160
	TFratios = []
	KLweight = []
	psnr = []
	with open('./{}/TFratio_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			TFratios.append(float(s))

	with open('./{}/KLweight_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			KLweight.append(float(s))

	with open('./{}/psnr_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			psnr.append(float(s))

	df = pd.DataFrame({
		'iters': np.linspace(1, n_iter, n_iter),
		'TFratios': TFratios,
		'KLweight': KLweight,
		'psnr': psnr})

	# plot score/ratio
	ax = df.plot(
		label='PSNR',
		style=['.'],
		kind='line',
		x='iters',
		y='psnr')
	ax2 = df.plot(
		label='TF ratio',
		style=['--'],
		kind='line',
		x='iters',
		y='TFratios',
		secondary_y=True,
		ax=ax)
	ax2 = df.plot(
		label='KL weight',
		style=['--'],
		kind='line',
		x='iters',
		y='KLweight',
		secondary_y=True,
		ax=ax)

	ax.set_ylabel('score')
	ax2.set_ylabel('ratio')
	plt.title('Training score/ratio curve')
	ax.get_legend().remove()
	lines_1, labels_1 = ax.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	lines = lines_1 + lines_2
	labels = labels_1 + labels_2
	plt.legend(lines, labels, loc='lower right', fancybox=True, framealpha=1, shadow=False, borderpad=1)
	plt.grid()
	plt.tight_layout()
	plt.show()

def plot_loss(log_dir):
	n_iter = 160
	n_iter=[i for i in range(n_iter)]
	loss=[]
	MSEloss=[]
	KLDloss=[]

	with open('./{}/loss_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			loss.append(float(s))

	with open('./{}/MSEloss_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			MSEloss.append(float(s))

	with open('./{}/KLDloss_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			KLDloss.append(float(s))

	plt.title('Training loss curve (learning curve)', fontsize=18)
	plt.plot(n_iter, loss, '-', linewidth=2, label="loss")
	plt.plot(n_iter, MSEloss, '-', linewidth=2, label="mse loss")
	plt.plot(n_iter, KLDloss, '-', linewidth=2, label="kld loss")
	plt.xlabel('iters', fontsize=12) 
	plt.ylabel('loss', fontsize=12)
	plt.ylim(-0.001,0.02) 
	plt.legend(loc = "upper right", fontsize=12)
	plt.show()

if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
	args = parser.parse_args()
	plot_psnr(args.log_dir)
	plot_score(args.log_dir)
	plot_loss(args.log_dir)