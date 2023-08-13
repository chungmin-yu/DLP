import matplotlib.pyplot as plt
import argparse

def plot_loss(log_dir):
	n_iter = 300
	n_iter=[i for i in range(n_iter)]
	loss=[]

	with open('./{}/loss_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			loss.append(float(s))

	plt.title('Training loss curve (learning curve)', fontsize=18)
	plt.plot(n_iter, loss, '-', linewidth=2, label="loss")
	plt.xlabel('epoch', fontsize=12) 
	plt.ylabel('loss', fontsize=12)
	plt.legend(loc = "upper right", fontsize=12)
	plt.show()

def plot_score(log_dir):
	n_iter = 300
	n_iter=[i for i in range(0, n_iter, 5)]
	score1=[]
	score2=[]

	with open('./{}/score1_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			score1.append(float(s))
	with open('./{}/score2_record.txt'.format(log_dir), 'r') as f:
		for line in f.readlines():
			s = line.strip('\n')
			score2.append(float(s))

	plt.title('F1-score curve', fontsize=18)
	plt.plot(n_iter, score1, '-', linewidth=2, label="test.json")
	plt.plot(n_iter, score2, '-', linewidth=2, label="new_test.json")
	plt.xlabel('epoch', fontsize=12) 
	plt.ylabel('score', fontsize=12)
	plt.legend(loc = "lower right", fontsize=12)
	plt.show()

if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', default='./checkpoint', help='base directory to save logs')
	args = parser.parse_args()
	plot_loss(args.log_dir)
	plot_score(args.log_dir)