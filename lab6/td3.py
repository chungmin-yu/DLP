'''DLP TD3 Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# TODO
# gym version: 0.26.2
### TD3 ###
# 1. Clipped Double Q-Learning for Actor-Critic
# 2. Delayed Policy Updates
# 3. Target Policy Smoothing Regularization
### TD3 ###
# select_action: u(s) + noise to select action (deterministic)
# ActorNet: create Actor network
# TD3 init: set optimizer
# sample: finish sample function 
# train: record.txt ->
# call update  -> update_behavior_network : update Q(s,a) from Q(s', a'), a' is from target actor network 
#											& set loss function of critic and actor network
#              -> _update_target_network : soft target update  
# test: take action & calculate total reward


class GaussianNoise:
	def __init__(self, dim, mu=None, std=None):
		self.mu = mu if mu else np.zeros(dim)
		self.std = np.ones(dim) * std if std else np.ones(dim) * .1

	def sample(self):
		return np.random.normal(self.mu, self.std)


class ReplayMemory:
	__slots__ = ['buffer']

	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)

	def __len__(self):
		return len(self.buffer)

	def append(self, *transition):
		# (state, action, reward, next_state, done)
		self.buffer.append(tuple(map(tuple, transition)))

	def sample(self, batch_size, device):
		'''sample a batch of transition tensors'''
		## TODO ##
		transitions = random.sample(self.buffer, batch_size)
		return (torch.tensor(x, dtype=torch.float, device=device)
				for x in zip(*transitions))


class ActorNet(nn.Module):
	def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
		super().__init__()
		## TODO ##
		h1, h2 = hidden_dim
		self.fc1 = nn.Linear(state_dim, h1)
		self.fc2 = nn.Linear(h1, h2)
		self.fc3 = nn.Linear(h2, action_dim)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, x):
		## TODO ##
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.tanh(x)

		return x


class CriticNet(nn.Module):
	def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
		super().__init__()
		h1, h2 = hidden_dim
		self.critic_head = nn.Sequential(
			nn.Linear(state_dim + action_dim, h1),
			nn.ReLU(),
		)
		self.critic = nn.Sequential(
			nn.Linear(h1, h2),
			nn.ReLU(),
			nn.Linear(h2, 1),
		)

	def forward(self, x, action):
		x = self.critic_head(torch.cat([x, action], dim=1))
		return self.critic(x)

class TD3:
	def __init__(self, args, max_action):
		# behavior network
		self._actor_net = ActorNet().to(args.device)
		self._critic_net1 = CriticNet().to(args.device)
		self._critic_net2 = CriticNet().to(args.device)
		# target network
		self._target_actor_net = ActorNet().to(args.device)
		self._target_critic_net1 = CriticNet().to(args.device)
		self._target_critic_net2 = CriticNet().to(args.device)
		# initialize target network
		self._target_actor_net.load_state_dict(self._actor_net.state_dict())
		self._target_critic_net1.load_state_dict(self._critic_net1.state_dict())
		self._target_critic_net2.load_state_dict(self._critic_net2.state_dict())
		## TODO ##
		self._actor_opt = torch.optim.Adam(self._actor_net.parameters(), lr=args.lra)
		self._critic_opt1 = torch.optim.Adam(self._critic_net1.parameters(), lr=args.lrc)
		self._critic_opt2 = torch.optim.Adam(self._critic_net2.parameters(), lr=args.lrc)
		# exploration noise
		self._exploration_noise = GaussianNoise(dim=2, std=args.exploration_noise)
		# policy noise
		self._policy_noise = GaussianNoise(dim=2, std=args.policy_noise)
		# memory
		self._memory = ReplayMemory(capacity=args.capacity)

		## config ##
		self.device = args.device
		self.batch_size = args.batch_size
		self.tau = args.tau
		self.gamma = args.gamma
		self.noise_clip = args.noise_clip
		self.freq = args.freq
		self.max_action = max_action

	def select_action(self, state, noise=True):
		'''based on the behavior (actor) network and exploration noise'''
		## TODO ##
		with torch.no_grad():
			if noise:
				sample_noise = torch.from_numpy(self._exploration_noise.sample()).view(1,-1).to(self.device)
				action = self._actor_net(torch.from_numpy(state).view(1,-1).to(self.device))
				action = (action + sample_noise).clamp(-1*self.max_action, self.max_action)
			else:
				# convert state to one row and feed on action net. convert tensor to 1-D numpy array.(via squeeze)
				action = self._actor_net(torch.from_numpy(state).view(1,-1).to(self.device))

		return action.cpu().numpy().squeeze()
		

	def append(self, state, action, reward, next_state, done):
		self._memory.append(state, action, [reward / 100], next_state,
							[int(done)])

	def update(self, total_steps):
		# update the behavior networks
		self._update_behavior_network(self.gamma, total_steps)
		# update the target networks
		if total_steps % self.freq == 0:
			self._update_target_network(self._target_actor_net, self._actor_net, self.tau)
			self._update_target_network(self._target_critic_net1, self._critic_net1, self.tau)
			self._update_target_network(self._target_critic_net2, self._critic_net2, self.tau)

	def _update_behavior_network(self, gamma, total_steps):
		actor_net, critic_net1, critic_net2, target_actor_net, target_critic_net1, target_critic_net2 = self._actor_net, self._critic_net1, self._critic_net2, self._target_actor_net, self._target_critic_net1, self._target_critic_net2
		actor_opt, critic_opt1, critic_opt2 = self._actor_opt, self._critic_opt1, self._critic_opt2

		# sample a minibatch of transitions
		state, action, reward, next_state, done = self._memory.sample(
			self.batch_size, self.device)

		## update critic ##
		# critic loss
		## TODO ##
		q_value1 = self._critic_net1(state, action)
		q_value2 = self._critic_net2(state, action)
		with torch.no_grad():
			sample_noise = torch.from_numpy(self._policy_noise.sample()).float().view(1,-1).to(self.device).clamp(-1*self.noise_clip, self.noise_clip)
			a_next = self._target_actor_net(next_state)
			a_next = (a_next + sample_noise).clamp(-1*self.max_action, self.max_action)
			q_next1 = self._target_critic_net1(next_state, a_next)
			q_next2 = self._target_critic_net2(next_state, a_next)
			q_target = reward + gamma * torch.min(q_next1, q_next2) * (1- done)   # final state: done=1
		
		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)
		critic_loss2 = criterion(q_value2, q_target)

		# optimize critic
		critic_net1.zero_grad()		
		critic_loss1.backward()		
		critic_opt1.step()

		critic_net2.zero_grad()
		critic_loss2.backward()
		critic_opt2.step()

		## TODO ##
		if total_steps % self.freq == 0:
			## update actor ##
			# actor loss
			# select action a from behavior actor network (a is different from sample transition's action)
			# get Q from behavior critic network, mean Q value -> objective function
			# maximize (objective function) = minimize -1 * (objective function)
			action = self._actor_net(state)
			actor_loss = -1 * (self._critic_net1(state, action).mean())
			# optimize actor
			actor_net.zero_grad()
			actor_loss.backward()
			actor_opt.step()

	@staticmethod
	def _update_target_network(target_net, net, tau):
		'''update target network by _soft_ copying from behavior network'''
		for target, behavior in zip(target_net.parameters(), net.parameters()):
			## TODO ##
			target.data.copy_((1 - tau) * target.data + tau * behavior.data)

	def save(self, model_path, checkpoint=False):
		if checkpoint:
			torch.save(
				{
					'actor': self._actor_net.state_dict(),
					'critic1': self._critic_net1.state_dict(),
					'critic2': self._critic_net2.state_dict(),
					'target_actor': self._target_actor_net.state_dict(),
					'target_critic1': self._target_critic_net1.state_dict(),
					'target_critic2': self._target_critic_net2.state_dict(),
					'actor_opt': self._actor_opt.state_dict(),
					'critic_opt1': self._critic_opt1.state_dict(),
					'critic_opt2': self._critic_opt2.state_dict(),
				}, model_path)
		else:
			torch.save(
				{
					'actor': self._actor_net.state_dict(),
					'critic1': self._critic_net1.state_dict(),
					'critic2': self._critic_net2.state_dict(),
				}, model_path)

	def load(self, model_path, checkpoint=False):
		model = torch.load(model_path)
		self._actor_net.load_state_dict(model['actor'])
		self._critic_net1.load_state_dict(model['critic1'])
		self._critic_net2.load_state_dict(model['critic2'])
		if checkpoint:
			self._target_actor_net.load_state_dict(model['target_actor'])
			self._target_critic_net1.load_state_dict(model['target_critic1'])
			self._target_critic_net2.load_state_dict(model['target_critic2'])
			self._actor_opt.load_state_dict(model['actor_opt'])
			self._critic_opt1.load_state_dict(model['critic_opt1'])
			self._critic_opt2.load_state_dict(model['critic_opt2'])


def train(args, env, agent, writer):
	print('Start Training')
	total_steps = 0
	ewma_reward = 0
	for episode in range(args.episode):
		total_reward = 0
		state = env.reset()
		state = np.array(state[0])
		for t in itertools.count(start=1):
			# select action
			if total_steps < args.warmup:
				action = env.action_space.sample()
			else:
				action = agent.select_action(state)
			# execute action
			next_state, reward, done, _, _ = env.step(action)
			# store transition
			agent.append(state, action, reward, next_state, done)
			if total_steps >= args.warmup:
				agent.update(total_steps)

			state = next_state
			total_reward += reward
			total_steps += 1
			if done:
				ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
				writer.add_scalar('Train/Episode Reward', total_reward,
								  total_steps)
				writer.add_scalar('Train/Ewma Reward', ewma_reward,
								  total_steps)
				print(
					'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
					.format(total_steps, episode, t, total_reward,
							ewma_reward))
				## TODO ##
				with open('td3_record.txt', 'a') as f:
					f.write(
					'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\n'
					.format(total_steps, episode, t, total_reward,
							ewma_reward))
				break
	env.close()


def test(args, env, agent, writer):
	print('Start Testing')
	seeds = (args.seed + i for i in range(10))
	rewards = []
	for n_episode, seed in enumerate(seeds):
		total_reward = 0
		state = env.reset(seed=seed)
		state = np.array(state[0])
		## TODO ##
		for t in itertools.count(start=1):
			# display the environment
			if args.render:
				env.render()

			# select action
			action = agent.select_action(state, False)
			# execute action
			next_state, reward, done, _, _ = env.step(action)
			# update state & total_reward
			state = next_state
			total_reward += reward

			# If achieve terminal state, record total reward
			if done:
				writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
				rewards.append(total_reward)
				print('episode {}: {:.2f}'.format(n_episode+1, total_reward))
				break 

	print('Average Reward', np.mean(rewards))
	env.close()


def main():
	## arguments ##
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-d', '--device', default='cuda')
	parser.add_argument('-m', '--model', default='td3.pth')
	parser.add_argument('--logdir', default='log/td3')
	# train
	parser.add_argument('--warmup', default=10000, type=int)
	parser.add_argument('--episode', default=1200, type=int)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--capacity', default=500000, type=int)
	parser.add_argument('--lra', default=1e-3, type=float)
	parser.add_argument('--lrc', default=1e-3, type=float)
	parser.add_argument('--gamma', default=.99, type=float)
	parser.add_argument('--tau', default=.005, type=float)
	# test
	parser.add_argument('--test_only', action='store_true')
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--seed', default=20200519, type=int)
	# td3
	parser.add_argument('--exploration_noise', default=0.1, type=float)
	parser.add_argument('--policy_noise', default=0.2, type=float)
	parser.add_argument('--noise_clip', default=0.5, type=float)
	parser.add_argument('--freq', default=2, type=int)
	args = parser.parse_args()

	## main ##
	if args.render:
		env = gym.make('LunarLanderContinuous-v2', render_mode='human')
	else:
		env = gym.make('LunarLanderContinuous-v2')
	max_action = float(env.action_space.high[0])
	agent = TD3(args, max_action)
	writer = SummaryWriter(args.logdir)
	if not args.test_only:
		train(args, env, agent, writer)
		agent.save(args.model)
	agent.load(args.model)
	test(args, env, agent, writer)


if __name__ == '__main__':
	main()
