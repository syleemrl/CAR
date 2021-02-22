from network2 import Actor
from network2 import Critic
from monitor import Monitor
import argparse
import random
import numpy as np
import tensorflow as tf
import pickle
import datetime
import os
import time
import sys
from IPython import embed
from copy import deepcopy
from utils import RunningMeanStd
from tensorflow.python import pywrap_tensorflow
import scipy.integrate as integrate
import types
np.set_printoptions(threshold=sys.maxsize)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
class PPO(object):
	def __init__(self, learning_rate_actor=2e-4, learning_rate_critic=0.001, learning_rate_decay=0.9993,
		gamma=0.95, gamma_sparse=0.99, lambd=0.95, epsilon=0.2):
		random.seed(int(time.time()))
		np.random.seed(int(time.time()))
		
		self.learning_rate_critic = learning_rate_critic
		self.learning_rate_actor = learning_rate_actor
		self.learning_rate_decay = learning_rate_decay
		self.epsilon = epsilon
		self.gamma = gamma
		self.gamma_sparse = gamma_sparse
		self.lambd = lambd
		self.reward_max = 0

	def initRun(self, pretrain, num_state, num_action, num_slaves=1):
		self.pretrain = pretrain

		self.num_slaves = num_slaves
		self.num_action = num_action
		self.num_state = num_state
		self.adaptive = False

		self.buildOptimize()

		if not parametric:
			self.ckpt = tf.train.Checkpoint(
				step=tf.Variable(0),
				actor_mean=self.actor.mean,
				actor_logstd=self.actor.logstd,
				critic=self.critic.value
			)
		else:
			self.ckpt = tf.train.Checkpoint(
				step=tf.Variable(0),
				actor_mean=self.actor.mean,
				actor_logstd=self.actor.logstd,
				critic=self.critic.value,
				critic_param=self.critic_param.value
			)

		if self.pretrain is not "":
			self.load(self.pretrain)
			li = pretrain.split("network")
			suffix = li[-1]
			self.RMS = RunningMeanStd(shape=(self.num_state))
			self.RMS.load(li[0]+"network"+li[1]+'rms'+suffix)
			self.RMS.setNumStates(self.num_state)

	def initTrain(self, name, env, parametric, pretrain="", evaluation=False, 
		directory=None, batch_size=1024, steps_per_iteration=10000, optim_frequency=5):

		self.name = name
		self.evaluation = evaluation
		self.directory = directory
		self.steps_per_iteration = [steps_per_iteration, steps_per_iteration * 0.5]
		self.optim_frequency = [optim_frequency, optim_frequency * 2]

		self.batch_size = batch_size
		self.batch_size_param = 128
		self.pretrain = pretrain
		self.parametric = parametric

		self.env = env
		self.num_slaves = self.env.num_slaves
		self.num_action = self.env.num_action
		self.num_state = self.env.num_state

		if self.parametric:
			self.num_param = self.env.dim_param

		self.param_x_batch = []
		self.param_y_batch = []

		self.buildOptimize()

		if not parametric:
			self.ckpt = tf.train.Checkpoint(
				actor_mean=self.actor.mean,
				actor_logstd=self.actor.logstd,
				critic=self.critic.value
			)
		else:
			self.ckpt = tf.train.Checkpoint(
				actor_mean=self.actor.mean,
				actor_logstd=self.actor.logstd,
				critic=self.critic.value,
				critic_param=self.critic_param.value
			)


		if self.pretrain is not "":
			self.load(self.pretrain)
			li = pretrain.split("network")
			suffix = li[-1]

			if len(li) == 2:
				self.env.RMS.load(li[0]+'rms'+suffix)
			else:
				self.env.RMS.load(li[0]+"network"+li[1]+'rms'+suffix)
			self.env.RMS.setNumStates(self.num_state)
		
		self.printSetting()
		

	def printSetting(self):
		
		print_list = []
		print_list.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print_list.append("test_name : {}".format(self.name))
		print_list.append("num_slaves : {}".format(self.num_slaves))
		print_list.append("num state : {}".format(self.num_state))
		print_list.append("num action : {}".format(self.num_action))
		print_list.append("learning_rate : {}".format(self.learning_rate_actor))
		print_list.append("gamma : {}".format(self.gamma))
		print_list.append("lambd : {}".format(self.lambd))
		print_list.append("batch_size : {}".format(self.batch_size))
		print_list.append("steps_per_iteration : {}".format(self.steps_per_iteration))
		print_list.append("clip ratio : {}".format(self.epsilon))
		print_list.append("pretrain : {}".format(self.pretrain))

		for s in print_list:
			print(s)

		if self.directory is not None:
			out = open(self.directory+"settings", "w")
			for s in print_list:
				out.write(s + "\n")
			out.close()

			out = open(self.directory+"results", "w")
			out.close()

	def buildOptimize(self):
		self.actor = Actor(self.num_state, self.num_action, 'actor')
		self.critic = Critic(self.num_state, 'critic')

		self.critic_trainer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_critic)
		self.actor_trainer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_actor)

		if self.parametric:
			self.critic_param = Critic(self.num_param, 'critic_param')

	@tf.function
	def train_critic_network(self, states, TD, param=False):
		if not param:
			critic = self.critic
		else:
			critic = self.critic_param

		with tf.GradientTape() as tape:
			loss = tf.reduce_mean(tf.square(critic.get_value(states) - TD))
		params = critic.get_variable(True)
		grads = tape.gradient(loss, params)
		if param:
			grads, _ = tf.clip_by_global_norm(grads, 0.5)
		
		self.critic_trainer.apply_gradients(zip(grads, params))
		return loss

	@tf.function
	def train_actor_network(self, states, actions, neglogp, GAE):
		with tf.GradientTape() as tape:
			means = self.actor.get_mean_action(states)
			cur_neglogp = self.actor.neglogp(actions, means)
			ratio = tf.exp(neglogp-cur_neglogp)
			clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
			loss = -tf.reduce_mean(tf.minimum(ratio*GAE, clipped_ratio*GAE))
		
		params = self.actor.get_variable(True)
		grads = tape.gradient(loss, params)
		grads, _ = tf.clip_by_global_norm(grads, 0.5)
		
		self.actor_trainer.apply_gradients(zip(grads, params))
		return loss

	def update(self, tuples):
		if not self.parametric:
			state_batch, action_batch, TD_batch, neglogp_batch, GAE_batch = self.compute_TD_GAE(tuples)
		else:
			state_batch, state_param_batch, action_batch, \
			TD_batch, TD_param_batch, \
			neglogp_batch, GAE_batch = self.compute_TD_GAE_parametric(tuples)

		if len(state_batch) < self.batch_size:
			return
		GAE_batch = (GAE_batch - GAE_batch.mean())/(GAE_batch.std() + 1e-5)

		ind = np.arange(len(state_batch))
		np.random.shuffle(ind)

		lossval_ac = 0
		lossval_c = 0
		for s in range(int(len(ind)//self.batch_size)):
			selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]
			lossval_ac += self.train_actor_network(tf.constant(state_batch[selectedIndex], dtype=np.float32), 
												   tf.constant(action_batch[selectedIndex], dtype=np.float32), 
												   tf.constant(neglogp_batch[selectedIndex], dtype=np.float32), 
												   tf.constant(GAE_batch[selectedIndex], dtype=np.float32))
			lossval_c += self.train_critic_network(tf.constant(state_batch[selectedIndex], dtype=np.float32), 
												   tf.constant(TD_batch[selectedIndex], dtype=np.float32))
		self.lossvals = []
		self.lossvals.append(['loss actor', lossval_ac])
		self.lossvals.append(['loss critic', lossval_c])
		
		if self.parametric:
			if len(self.param_x_batch) == 0:
				self.param_x_batch = state_param_batch
				self.param_y_batch = TD_param_batch
			else:
				if len(state_param_batch) != 0:
					self.param_x_batch = np.concatenate((self.param_x_batch, state_param_batch), axis=0)
					self.param_y_batch = np.concatenate((self.param_y_batch, TD_param_batch), axis=0)

				if len(self.param_x_batch) > 5000:
					self.param_x_batch = self.param_x_batch[-2000:]
					self.param_y_batch = self.param_y_batch[-2000:]
			if self.env.mode == 1 and self.env.mode_counter % 3 == 1:
				lossval_cp = 0

				for n in range(50):
					ind = np.arange(len(self.param_x_batch))
					np.random.shuffle(ind)
					for s in range(int(len(ind)//self.batch_size_param)):
						selectedIndex = ind[s*self.batch_size_param:(s+1)*self.batch_size_param]
						lossval_cp += self.train_critic_network(tf.constant(state_param_batch[selectedIndex], dtype=np.float32), 
																tf.constant(TD_param_batch[selectedIndex], dtype=np.float32),
																param=True)
				self.lossvals.append(['loss critic param', lossval_cp / 50])

	def compute_TD_GAE(self, tuples):
		state_batch = []
		action_batch = []
		TD_batch = []
		neglogp_batch = []
		GAE_batch = []

		for data in tuples:
			size = len(data)		
			# get values
			states, actions, rewards, values, neglogprobs, times = zip(*data)
			values = np.concatenate((values, [0]), axis=0)
			advantages = np.zeros(size)
			ad_t = 0
			for i in reversed(range(len(data))):
				if i == size - 1:
					timestep = 0
				elif times[i] > times[i+1]:
					timestep = self.env.phaselength - times[i]
				else:
					timestep = times[i+1]  - times[i]

				t = integrate.quad(lambda x: pow(self.gamma, x), 0, timestep)[0]
				delta = t * rewards[i] + values[i+1] * pow(self.gamma, timestep)  - values[i]
				ad_t = delta + pow(self.lambd, timestep)* pow(self.gamma, timestep)  * ad_t
				advantages[i] = ad_t

			TD = values[:size] + advantages
			for i in range(size):
				state_batch.append(states[i])
				action_batch.append(actions[i])
				TD_batch.append(TD[i])
				neglogp_batch.append(neglogprobs[i])
				GAE_batch.append(advantages[i])
		return np.array(state_batch), np.array(action_batch), np.array(TD_batch), np.array(neglogp_batch), np.array(GAE_batch)

	def compute_TD_GAE_parametric(self, tuples):
		state_batch = []
		state_param_batch = []
		action_batch = []
		TD_batch = []
		TD_param_batch = []
		neglogp_batch = []
		GAE_batch = []
		param_info_batch = []

		for data in tuples:
			# get values
			states, actions, rewards, values, neglogprobs, times, param, param_info = zip(*data)

			if len(times) == self.env.phaselength * 5 + 10 + 1:
				if times[-1] < self.env.phaselength - 1.8:
					for i in reversed(range(len(times))):
						if i != len(times) - 1 and times[i] > times[i + 1]:
							count = i
							break
					states = states[:count+1]
					actions = actions[:count+1]
					rewards = rewards[:count+1]
					values = values[:count+1]
					neglogprobs = neglogprobs[:count+1]
					times = times[:count+1]
					param = param[:count+1]
					param_info = param_info[:count+1]

			size = len(times)		


			values = np.concatenate((np.array(values), [0]), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			count_V = 0
			sum_V = 0
			V = 0
			flag = True
			for i in reversed(range(size)):
				if i == size - 1 or (i == size - 2 and times[i+1] == 0):
					timestep = 0
				elif times[i] > times[i+1]:
					timestep = self.env.phaselength - times[i] + times[i+1]
				else:
					timestep = times[i+1]  - times[i]
				
				t = integrate.quad(lambda x: pow(self.gamma, x), 0, timestep)[0]
				delta = t * rewards[i][0] + values[i+1] * pow(self.gamma, timestep) - values[i]
				V = t * rewards[i][0] + 4 * rewards[i][1] + V * pow(self.gamma, timestep)
				if rewards[i][1] != 0:
					delta += rewards[i][1]

				ad_t = delta + pow(self.lambd, timestep) * pow(self.gamma, timestep) * ad_t
				advantages[i] = ad_t


				sum_V += V
				count_V += 1
				if i != size - 1 and (i == 0 or times[i-1] > times[i]):
					if flag:
						param_info_batch.append(param_info[i])
						state_param_batch.append(param[i])
						TD_param_batch.append(sum_V / count_V)

					count_V = 0
					sum_V = 0
					V = 0
					
			TD = values[:size] + advantages

			for i in range(size):
				state_batch.append(states[i])
				action_batch.append(actions[i])
				TD_batch.append(TD[i])
				neglogp_batch.append(neglogprobs[i])
				GAE_batch.append(advantages[i])
		
		self.v_param = TD_param_batch
		self.info_param = param_info_batch
		return np.array(state_batch), np.array(state_param_batch), np.array(action_batch), \
			   np.array(TD_batch), np.array(TD_param_batch), \
			   np.array(neglogp_batch), np.array(GAE_batch)
					
	def save(self):
		self.ckpt.write(self.directory+'network-0')
		self.env.RMS.save(self.directory+'rms-0')

	def load(self, path):
		print("Loading parameters from {}".format(path))
		saved_variables = tf.train.list_variables(path)
		saved_values = []
		for v in saved_variables:
			saved_values.append(tf.train.load_variable(path,v[0]))
		saved_dict = {n[0] : v for n, v in zip(saved_variables, saved_values)}
		trainable_variables = self.actor.get_variable(True) + self.critic.get_variable(True)
		if self.parametric:
			trainable_variables += self.critic_param.get_variable(True)
		for v in trainable_variables:
			key = v.name[:-2]+'/.ATTRIBUTES/VARIABLE_VALUE'
			if key in saved_dict:
				saved_v = saved_dict[key]
				if v.shape == saved_v.shape:
					print("Restore {}".format(key))
					v.assign(saved_v)
				elif "L1/kernel" in v.name and v.shape[0] > saved_v.shape[0]:
					l = v.shape[0] - saved_v.shape[0]
					new_v = np.zeros((l, v.shape[1]), dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=0)
					v.assign(saved_v)
					print("Restore {}, add {} input nodes".format(key, l))

				elif ("mean/bias" in v.name or "std" in v.name) and v.shape[0] > saved_v.shape[0]:
					l = v.shape[0] - saved_v.shape[0]
					new_v = np.zeros(l, dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=0)
					v.assign(saved_v)
					print("Restore {}, add {} output nodes".format(key, l))

				elif "mean/kernel" in v.name and v.shape[1] > saved_v.shape[1]:
					l = v.shape[1] - saved_v.shape[1]
					new_v = np.zeros((v.shape[0], l), dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=1)
					v.assign(saved_v)
					print("Restore {}, add {} output nodes".format(key, l))

	def printNetworkSummary(self):
		print_list = []
		for v in self.lossvals:
			print_list.append('{}: {:.3f}'.format(v[0], v[1]))

		print_list.append('===============================================================')
		for s in print_list:
			print(s)


	def train(self, num_iteration):
		epi_info_iter = []
		epi_info_iter_hind = []

		self.env.sampler.resetExplore()
		it_cur = 0

		for it in range(num_iteration):
			for i in range(self.num_slaves):
				self.env.reset(i)
			states = self.env.getStates()
			local_step = 0
			last_print = 0
	
			epi_info = [[] for _ in range(self.num_slaves)]	

			if self.parametric:
				param_info = self.env.updateGoal(self.critic_param)
			else:
				param_info = -1

			while True:
				# set action
				actions, neglogprobs = self.actor.get_action(states)
				actions = actions.numpy()
				neglogprobs = neglogprobs.numpy()
				values = self.critic.get_value(states)

				rewards, dones, times, params = self.env.step(actions)
				for j in range(self.num_slaves):
					if not self.env.getTerminated(j):
						if not self.parametric and rewards[j] is not None:
							epi_info[j].append([states[j], actions[j], rewards[j], values[j], neglogprobs[j], times[j]])
							local_step += 1
						if self.parametric and rewards[j][0] is not None:
							epi_info[j].append([states[j], actions[j], rewards[j], values[j], neglogprobs[j], times[j], params[j], param_info])
							local_step += 1
						if dones[j]:
							if len(epi_info[j]) != 0:
								epi_info_iter.append(deepcopy(epi_info[j]))
							
							if local_step < self.steps_per_iteration[self.parametric]:
								epi_info[j] = []
								self.env.reset(j)
							else:
								self.env.setTerminated(j)
				if local_step >= self.steps_per_iteration[self.parametric]:
					if self.env.getAllTerminated():
						print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration[self.parametric]),end='\r')
						break
				if last_print + 100 < local_step: 
					print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration[self.parametric]),end='\r')
					last_print = local_step
				
				states = self.env.getStates()
			if self.parametric:
				self.env.sampler.saveProgress(self.env.mode)
			it_cur += 1

			print('')

			if (self.env.mode < 2 and it_cur % self.optim_frequency[self.parametric] == self.optim_frequency[self.parametric] - 1) or \
			    (self.env.mode == 2 and it_cur % self.env.sampler.eval_frequency == self.env.sampler.eval_frequency - 1):	
				self.update(epi_info_iter) 

				if self.parametric:
					t = self.env.updateCurriculum(self.critic_param, self.v_param, self.info_param)
					if t == -1:
						break
					elif t == 1:
						it_cur = 0

				if self.learning_rate_actor > 1e-5:
					self.learning_rate_actor = self.learning_rate_actor * self.learning_rate_decay

				summary = self.env.printSummary()
				self.printNetworkSummary()
				if self.directory is not None:
					self.save()

				if self.directory is not None and self.reward_max < summary['r_per_e']:
					self.reward_max = summary['r_per_e']
					self.env.RMS.save(self.directory+'rms-rmax')

					os.system("cp {}network-{}.data-00000-of-00001 {}network-rmax.data-00000-of-00001".format(self.directory, 0, self.directory))
					os.system("cp {}network-{}.index {}network-rmax.index".format(self.directory, 0, self.directory))

				epi_info_iter = []

	def run(self, state):
		state = np.reshape(state, (1, self.num_state))
		state = self.RMS.apply(state)
		
		values = self.critic.get_value(state)
		action = self.actor.get_mean_action(state)

		return action

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ntimesteps", type=int, default=1000000)
	parser.add_argument("--ref", type=str, default="")
	parser.add_argument("--test_name", type=str, default="")
	parser.add_argument("--pretrain", type=str, default="")
	parser.add_argument("--nslaves", type=int, default=4)
	parser.add_argument("--parametric", dest='parametric', action='store_true')
	parser.add_argument("--adaptive", dest='adaptive', action='store_true')
	parser.add_argument("--save", type=bool, default=True)
	parser.add_argument("--no-plot", dest='plot', action='store_false')
	parser.set_defaults(plot=True)
	parser.set_defaults(adaptive=False)
	parser.set_defaults(parametric=False)

	args = parser.parse_args()

	directory = None
	if args.save:
		if not os.path.exists("./output/"):
			os.mkdir("./output/")

		directory = "./output/" + args.test_name + "/"
		if not os.path.exists(directory):
			os.mkdir(directory)

	if args.pretrain != "":
		env = Monitor(ref=args.ref, num_slaves=args.nslaves, directory=directory, plot=args.plot, adaptive=args.adaptive, parametric=args.parametric)
	else:
		env = Monitor(ref=args.ref, num_slaves=args.nslaves, directory=directory, plot=args.plot, adaptive=args.adaptive, parametric=args.parametric)

	ppo = PPO()

	ppo.initTrain(env=env, name=args.test_name, directory=directory, pretrain=args.pretrain, parametric=args.parametric)

	ppo.train(args.ntimesteps)