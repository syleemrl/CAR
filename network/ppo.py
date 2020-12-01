import warnings
warnings.filterwarnings(action='ignore') 
from network import Actor
from network import Critic
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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
	tf.contrib._warning = None
class PPO(object):
	def __init__(self, learning_rate_actor=2e-4, learning_rate_critic=0.001, learning_rate_decay=0.9993,
		gamma=0.95, gamma_sparse=0.99, lambd=0.95, epsilon=0.2):
		random.seed(int(time.time()))
		np.random.seed(int(time.time()))
		tf.set_random_seed(int(time.time()))
		
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

		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = self.num_slaves
		config.inter_op_parallelism_threads = self.num_slaves
		self.sess = tf.Session(config=config)
		self.adaptive = False

		#build network and optimizer
		name = pretrain.split("/")[-2]
		self.buildOptimize(name)
		
		save_list = [v for v in tf.trainable_variables() if v.name.find(name)!=-1]
		self.saver = tf.train.Saver(var_list=save_list, max_to_keep=1)
		
		self.step = 0

		if self.pretrain is not "":
			self.load(self.pretrain)
			li = pretrain.split("network")
			suffix = li[-1]
			self.RMS = RunningMeanStd(shape=(self.num_state))
			self.RMS.load(li[0]+"network"+li[1]+'rms'+suffix)
			self.RMS.setNumStates(self.num_state)

	def initTrain(self, name, env, adaptive, parametric, pretrain="", evaluation=False, 
		directory=None, batch_size=1024, steps_per_iteration=10000, optim_frequency=5):

		self.name = name
		self.evaluation = evaluation
		self.directory = directory
		self.steps_per_iteration = [steps_per_iteration * 0.25, steps_per_iteration * 0.25,  steps_per_iteration * 0.25]
		self.optim_frequency = [optim_frequency * 4, optim_frequency * 4, optim_frequency * 4]

		self.batch_size = batch_size
		self.batch_size_target = 128
		self.pretrain = pretrain
		self.adaptive = adaptive
		self.parametric = parametric
		self.env = env
		self.num_slaves = self.env.num_slaves
		self.num_action = self.env.num_action
		self.num_state = self.env.num_state

		self.target_x_batch = []
		self.target_y_batch = []

		self.last_target_update = 0

		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = self.num_slaves
		config.inter_op_parallelism_threads = self.num_slaves
		self.sess = tf.Session(config=config)

		#build network and optimizer
		self.buildOptimize(name)
		# load pretrained network
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
			out = open(self.directory+"parameters", "w")
			for s in print_list:
				out.write(s + "\n")
			out.close()

			out = open(self.directory+"results", "w")
			out.close()

	def createCriticNetwork(self, name, input, TD, clip=True):
		critic = Critic(self.sess, name, input)
			
		with tf.variable_scope(name+'_Optimize'):
			value_loss = tf.reduce_mean(tf.square(critic.value - TD))
			loss = value_loss

		critic_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
		grads, params = zip(*critic_trainer.compute_gradients(loss))
		if clip:
			grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		critic_train_op = critic_trainer.apply_gradients(grads_and_vars)
			
		return critic, critic_train_op, loss

	def buildOptimize(self, name):
		self.state = tf.placeholder(tf.float32, shape=[None, self.num_state], name=name+'_state')
		self.actor = Actor(self.sess, name, self.state, self.num_action)

		with tf.variable_scope(name+'_Optimize'):
			self.action = tf.placeholder(tf.float32, shape=[None,self.num_action], name='action')
			self.TD = tf.placeholder(tf.float32, shape=[None], name='TD')
			self.GAE = tf.placeholder(tf.float32, shape=[None], name='GAE')
			self.old_neglogp = tf.placeholder(tf.float32, shape=[None], name='old_neglogp')
			self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate')
			self.cur_neglogp = self.actor.neglogp(self.action)
			self.ratio = tf.exp(self.old_neglogp-self.cur_neglogp)
			clipped_ratio = tf.clip_by_value(self.ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

			surrogate = -tf.reduce_mean(tf.minimum(self.ratio*self.GAE, clipped_ratio*self.GAE))
			self.loss_actor = surrogate

		actor_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
		grads, params = zip(*actor_trainer.compute_gradients(self.loss_actor));
		grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		self.actor_train_op = actor_trainer.apply_gradients(grads_and_vars)

		self.critic, self.critic_train_op, self.loss_critic = self.createCriticNetwork(name, self.state, self.TD)

		if self.adaptive:
			if self.parametric:
				self.state_target = tf.placeholder(tf.float32, shape=[None, self.env.dim_param], name=name+'_state_target')

			with tf.variable_scope(name+'_Optimize'):
				if self.parametric:
					self.TD_target = tf.placeholder(tf.float32, shape=[None], name='TD_target')

			if self.parametric:
				self.critic_target, self.critic_target_train_op, self.loss_critic_target = self.createCriticNetwork(name+'_target', self.state_target, self.TD_target, False)

		var_list = tf.trainable_variables()
		save_list = []
		for v in var_list:
			if "Actor" in v.name or "Critic" in v.name:
				save_list.append(v)

		self.saver = tf.train.Saver(var_list=save_list, max_to_keep=1)
		
		self.sess.run(tf.global_variables_initializer())


	def update(self, tuples):
		state_batch, action_batch, TD_batch, neglogp_batch, GAE_batch = self.computeTDandGAE(tuples)
		if len(state_batch) < self.batch_size:
			return
		GAE_batch = (GAE_batch - GAE_batch.mean())/(GAE_batch.std() + 1e-5)

		ind = np.arange(len(state_batch))
		np.random.shuffle(ind)

		lossval_ac = 0
		lossval_c = 0
		for s in range(int(len(ind)//self.batch_size)):
			selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]
			val = self.sess.run([self.actor_train_op, self.critic_train_op, 
								self.loss_actor, self.loss_critic], 
				feed_dict={
					self.state: state_batch[selectedIndex], 
					self.TD: TD_batch[selectedIndex], 
					self.action: action_batch[selectedIndex], 
					self.old_neglogp: neglogp_batch[selectedIndex], 
					self.GAE: GAE_batch[selectedIndex],
					self.learning_rate_ph:self.learning_rate_actor
				}
			)
			lossval_ac += val[2]
			lossval_c += val[3]
		self.lossvals = []
		self.lossvals.append(['loss actor', lossval_ac])
		self.lossvals.append(['loss critic', lossval_c])

	def computeTDandGAE(self, tuples):
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

	def updateAdaptive(self, tuples):
		state_batch, state_target_batch, action_batch, \
		TD_batch, TD_target_batch, \
		neglogp_batch, GAE_batch = self.computeTDandGAEAdaptive(tuples)


		if len(state_batch) < self.batch_size:
			return
		GAE_batch = (GAE_batch - GAE_batch.mean())/(GAE_batch.std() + 1e-5)

		ind = np.arange(len(state_batch))
		np.random.shuffle(ind)

		lossval_ac = 0
		lossval_c = 0
		lossval_ct = 0

		for s in range(int(len(ind)//self.batch_size)):
			selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]
			val = self.sess.run([self.actor_train_op, self.critic_train_op,
							self.loss_actor, self.loss_critic], 
				feed_dict={
					self.state: state_batch[selectedIndex], 
					self.TD: TD_batch[selectedIndex], 
					self.action: action_batch[selectedIndex], 
					self.old_neglogp: neglogp_batch[selectedIndex], 
					self.GAE: GAE_batch[selectedIndex],
					self.learning_rate_ph:self.learning_rate_actor
				}
			)
			lossval_ac += val[2]
			lossval_c += val[3]
		
		self.lossvals = []

		self.lossvals.append(['loss actor', lossval_ac])
		self.lossvals.append(['loss critic', lossval_c])
		if self.parametric:
			if len(self.target_x_batch) == 0:
				self.target_x_batch = state_target_batch
				self.target_y_batch = TD_target_batch
			else:
				self.target_x_batch = np.concatenate((self.target_x_batch, state_target_batch), axis=0)
				self.target_y_batch = np.concatenate((self.target_y_batch, TD_target_batch), axis=0)

				if len(self.target_x_batch) > 5000:
					self.target_x_batch = self.target_x_batch[-2000:]
					self.target_y_batch = self.target_y_batch[-2000:]
			for n in range(50):
				ind = np.arange(len(self.target_x_batch))
				np.random.shuffle(ind)
				for s in range(int(len(ind)//self.batch_size_target)):
					selectedIndex = ind[s*self.batch_size_target:(s+1)*self.batch_size_target]
					val = self.sess.run([self.critic_target_train_op, self.loss_critic_target], 
						feed_dict={
							self.state_target: self.target_x_batch[selectedIndex], 
							self.TD_target: self.target_y_batch[selectedIndex]
						}
					)
					lossval_ct += val[1]
			self.lossvals.append(['loss critic target', lossval_ct / 50])

	def computeTDandGAEAdaptive(self, tuples):
		state_batch = []
		state_target_batch = []
		action_batch = []
		TD_batch = []
		TD_target_batch = []
		neglogp_batch = []
		GAE_batch = []
		idx_batch = []

		for data in tuples:
			# get values
			states, actions, rewards, values, neglogprobs, times, param, idx = zip(*data)

			if len(times) == self.env.phaselength * 3 + 10 + 1:
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
					idx = idx[:count+1]

			size = len(times)		


			values = np.concatenate((np.array(values), [0]), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			# TD_t_dense = 0
			# TD_t_sparse = 0
			count_V = 0
			sum_V = 0
			V = 0
			flag = False
			for i in reversed(range(size)):
				if i == size - 1 or (i == size - 2 and times[i+1] == 0):
					timestep = 0
				elif times[i] > times[i+1]:
					timestep = self.env.phaselength - times[i] + times[i+1]
				else:
					timestep = times[i+1]  - times[i]
				
				t = integrate.quad(lambda x: pow(self.gamma, x), 0, timestep)[0]
				delta = t * rewards[i][0] + values[i+1] * pow(self.gamma, timestep) - values[i]
				V = t * rewards[i][0] + 1 * rewards[i][1] + V * pow(self.gamma, timestep)
				if rewards[i][1] != 0:
					delta += rewards[i][1]
					flag = True
				ad_t = delta + pow(self.lambd, timestep) * pow(self.gamma, timestep) * ad_t
				advantages[i] = ad_t

				# TD_t_dense = rewards[i][0] + TD_t_dense
				# TD_t_sparse = rewards[i][1] + TD_t_sparse
				sum_V += V
				count_V += 1
				if i != size - 1 and (i == 0 or times[i-1] > times[i]):
					# if TD_t_sparse != 0:
					# 	idx_batch.append(idx[i])
					# 	state_target_batch.append(param[i])
					# 	TD_target_batch.append(1 / self.env.phaselength * TD_t_dense + 1.0 / 10.0 * TD_t_sparse)

					# TD_t_dense = 0
					# TD_t_sparse = 0
					if flag:
						idx_batch.append(idx[i])
						state_target_batch.append(param[i])
						TD_target_batch.append(sum_V / count_V)

					count_V = 0
					sum_V = 0
					V = 0
					flag = False
					
			TD = values[:size] + advantages

			for i in range(size):
				state_batch.append(states[i])
				action_batch.append(actions[i])
				TD_batch.append(TD[i])
				neglogp_batch.append(neglogprobs[i])
				GAE_batch.append(advantages[i])
		
		self.v_target = TD_target_batch
		self.idx_target = idx_batch
		return np.array(state_batch), np.array(state_target_batch), np.array(action_batch), \
			   np.array(TD_batch), np.array(TD_target_batch), \
			   np.array(neglogp_batch), np.array(GAE_batch)

	def computeValue(self, tuples):
		marginal_vs = []

		for data in tuples:
			rewards, times = zip(*data)

			size = len(rewards)

			count_V = 0
			sum_V = 0
			V = 0
			flag = False
			for i in reversed(range(len(rewards))):
				if i == size - 1 or (i == size - 2 and times[i+1] == 0):
					timestep = 0
				elif times[i] > times[i+1]:
					timestep = self.env.phaselength - times[i] + times[i+1]
				else:
					timestep = times[i+1]  - times[i]
				
				t = integrate.quad(lambda x: pow(self.gamma, x), 0, timestep)[0]
				V = t * rewards[i][0] + rewards[i][1] + V * pow(self.gamma, timestep)
				if rewards[i][1] != 0:
					flag = True

				sum_V += V
				count_V += 1
				if i != size - 1 and (i == 0 or times[i-1] > times[i]):
					if flag:
						marginal_vs.append(sum_V / count_V)

					count_V = 0
					sum_V = 0
					V = 0
					flag = False

		return marginal_vs
					
	def save(self):
		self.saver.save(self.sess, self.directory + "network", global_step = 0)
		self.env.RMS.save(self.directory+'rms-0')

	def load(self, path):
		print("Loading parameters from {}".format(path))

		def get_tensors_in_checkpoint_file(file_name):
			varlist=[]
			var_value =[]
			reader = pywrap_tensorflow.NewCheckpointReader(file_name)
			var_to_shape_map = reader.get_variable_to_shape_map()
			for key in sorted(var_to_shape_map):
				varlist.append(key)
				var_value.append(reader.get_tensor(key))
			return (varlist, var_value)

		saved_variables, saved_values = get_tensors_in_checkpoint_file(path)
		saved_dict = {n : v for n, v in zip(saved_variables, saved_values)}
		restore_op = []
		for v in tf.trainable_variables():
			if v.name[:-2] in saved_dict:
				saved_v = saved_dict[v.name[:-2]]
				if v.shape == saved_v.shape:
					print("Restore {}".format(v.name[:-2]))
					restore_op.append(v.assign(saved_v))
				elif "L1/kernel" in v.name and v.shape[0] > saved_v.shape[0]:
					l = v.shape[0] - saved_v.shape[0]
					new_v = np.zeros((l, v.shape[1]), dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=0)
					restore_op.append(v.assign(saved_v))
					print("Restore {}, add {} input nodes".format(v.name[:-2], l))

				elif ("mean/bias" in v.name or "std" in v.name) and v.shape[0] > saved_v.shape[0]:
					l = v.shape[0] - saved_v.shape[0]
					new_v = np.zeros(l, dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=0)
					restore_op.append(v.assign(saved_v))
					print("Restore {}, add {} output nodes".format(v.name[:-2], l))

				elif "mean/kernel" in v.name and v.shape[1] > saved_v.shape[1]:
					l = v.shape[1] - saved_v.shape[1]
					new_v = np.zeros((v.shape[0], l), dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=1)
					restore_op.append(v.assign(saved_v))
					print("Restore {}, add {} output nodes".format(v.name[:-2], l))

		restore_op = tf.group(*restore_op)
		self.sess.run(restore_op)


	def printNetworkSummary(self):
		print_list = []
		print_list.append('noise : {:.3f}'.format(self.sess.run(self.actor.std).mean()))
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
			if self.adaptive:
				p_idx = self.env.updateGoal(self.critic_target)
			while True:
				# set action
				actions, neglogprobs = self.actor.getAction(states)

				values = self.critic.getValue(states)

				rewards, dones, times, params = self.env.step(actions)
				for j in range(self.num_slaves):
					if not self.env.getTerminated(j):
						if not self.adaptive and rewards[j] is not None:
							epi_info[j].append([states[j], actions[j], rewards[j], values[j], neglogprobs[j], times[j]])
							local_step += 1
						if self.adaptive and rewards[j][0] is not None:
							epi_info[j].append([states[j], actions[j], rewards[j], values[j], neglogprobs[j], times[j], params[j], p_idx])
							local_step += 1
						if dones[j]:
							if len(epi_info[j]) != 0:
								epi_info_iter.append(deepcopy(epi_info[j]))
							
							if local_step < self.steps_per_iteration[self.env.mode]:
								epi_info[j] = []
								self.env.reset(j)
							else:
								self.env.setTerminated(j)

				if local_step >= self.steps_per_iteration[self.env.mode]:
					if self.env.getAllTerminated():
						print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration[self.env.mode]),end='\r')
						break
				if last_print + 100 < local_step: 
					print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration[self.env.mode]),end='\r')
					last_print = local_step
				
				states = self.env.getStates()
			self.env.sampler.saveProgress(self.env.mode)
			it_cur += 1

			print('')

			if (self.env.mode < 2 and it_cur % self.optim_frequency[self.env.mode] == self.optim_frequency[self.env.mode] - 1) or \
			    (self.env.mode == 2 and it_cur % self.env.sampler.eval_frequency == self.env.sampler.eval_frequency - 1):	
				if self.adaptive:
					self.updateAdaptive(epi_info_iter)
					t = self.env.updateMode(self.critic_target, self.v_target)
					if t == -1:
						break
					elif t == 1:
						it_cur = 0
					
					if not self.parametric:
						self.env.updateAdaptive()
					else:
						self.env.updateCurriculum(self.critic_target)
					epi_info_iter_hind = []

					if self.env.needEvaluation():
						self.eval(30)

				else:			
					self.update(epi_info_iter) 

				if self.learning_rate_actor > 1e-5:
					self.learning_rate_actor = self.learning_rate_actor * self.learning_rate_decay

				summary = self.env.printSummary()
				self.printNetworkSummary()
				if self.directory is not None:
					self.save()

				if self.directory is not None and self.reward_max < summary['r_per_e']:
					self.reward_max = summary['r_per_e']
					self.env.RMS.save(self.directory+'rms-rmax')

					os.system("cp {}/network-{}.data-00000-of-00001 {}/network-rmax.data-00000-of-00001".format(self.directory, 0, self.directory))
					os.system("cp {}/network-{}.index {}/network-rmax.index".format(self.directory, 0, self.directory))
					os.system("cp {}/network-{}.meta {}/network-rmax.meta".format(self.directory, 0, self.directory))

				epi_info_iter = []

	def eval(self, num_samples):
		tuples = []
		for it in range(num_samples):
			for i in range(self.num_slaves):
				self.env.reset(i)
			states = self.env.getStates()
			local_step = 0
			tuples_iter = [[] for _ in range(self.num_slaves)]	
	
			self.env.updateGoal(self.critic_target, False)
			while True:
				# set action
				actions, neglogprobs = self.actor.getAction(states)
				values = self.critic.getValue(states)

				rewards, dones, times, params = self.env.step(actions, False)
				for j in range(self.num_slaves):
					if not self.env.getTerminated(j):
						if rewards[j][0] is not None:
							tuples_iter[j].append([rewards[j], times[j]])
							local_step += 1
						if dones[j]:
							if len(tuples_iter[j]) != 0:
								tuples.append(deepcopy(tuples_iter[j]))
							
							if local_step < 1000:
								tuples_iter[j] = []
								self.env.reset(j)
							else:
								self.env.setTerminated(j)
				if self.env.getAllTerminated():
					break
				states = self.env.getStates()
		marginal_vs = self.computeValue(tuples)
		self.env.saveEvaluation(marginal_vs)
	def run(self, state):
		state = np.reshape(state, (1, self.num_state))
		state = self.RMS.apply(state)
		
		values = self.critic.getValue(state)
		#action, _ = self.actor.getAction(state)
		action = self.actor.getMeanAction(state)

		return action

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ntimesteps", type=int, default=1000000)
	parser.add_argument("--ref", type=str, default="")
	parser.add_argument("--test_name", type=str, default="")
	parser.add_argument("--pretrain", type=str, default="")
	parser.add_argument("--nslaves", type=int, default=4)
	parser.add_argument("--adaptive", dest='adaptive', action='store_true')
	parser.add_argument("--parametric", dest='parametric', action='store_true')
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

	ppo.initTrain(env=env, name=args.test_name, directory=directory, pretrain=args.pretrain, 
		adaptive=args.adaptive, parametric=args.parametric)

	ppo.train(args.ntimesteps)
