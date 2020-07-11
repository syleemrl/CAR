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
import types
np.set_printoptions(threshold=sys.maxsize)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
	tf.contrib._warning = None
class PPO(object):
	def __init__(self, learning_rate_actor=2e-4, learning_rate_critic=0.001, learning_rate_decay=0.9993,
		gamma=0.95, gamma_sparse=0.97, lambd=0.95, epsilon=0.2):
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

	def initTrain(self, name, env, adaptive, pretrain="", evaluation=False, 
		directory=None, batch_size=1024, steps_per_iteration=10000):

		self.name = name
		self.evaluation = evaluation
		self.directory = directory
		self.steps_per_iteration = steps_per_iteration
		self.batch_size = batch_size
		self.pretrain = pretrain
		self.adaptive = adaptive
		self.env = env
		self.num_slaves = self.env.num_slaves
		self.num_action = self.env.num_action
		self.num_state = self.env.num_state

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
			self.env.RMS.load(li[0]+'rms'+suffix)
			self.env.RMS.setNumStates(self.num_state)
			self.env.sim_env.LoadAdaptiveMotion()
		
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

	def buildOptimize(self, name):
		self.state = tf.placeholder(tf.float32, shape=[None, self.num_state], name=name+'_state')
		self.actor = Actor(self.sess, name, self.state, self.num_action)
		self.critic = Critic(self.sess, name, self.state)

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
			value_loss = tf.reduce_mean(tf.square(self.critic.value - self.TD))
			self.loss_actor = surrogate
			self.loss_critic = value_loss

		actor_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
		grads, params = zip(*actor_trainer.compute_gradients(self.loss_actor));
		grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		self.actor_train_op = actor_trainer.apply_gradients(grads_and_vars)

		critic_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
		grads, params = zip(*critic_trainer.compute_gradients(self.loss_critic));
		grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		self.critic_train_op = critic_trainer.apply_gradients(grads_and_vars)

		if self.adaptive:
			self.critic_sparse = Critic(self.sess, name, self.state, postfix='_sparse')
			
			with tf.variable_scope(name+'_Optimize'):
				self.TD_sparse = tf.placeholder(tf.float32, shape=[None], name='TD')
				value_sparse_loss = tf.reduce_mean(tf.square(self.critic_sparse.value - self.TD_sparse))
				self.loss_critic_sparse = value_sparse_loss

			critic_trainer_sparse = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
			grads, params = zip(*critic_trainer_sparse.compute_gradients(self.loss_critic_sparse));
			grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
			grads_and_vars = list(zip(grads, params))
			self.critic_sparse_train_op = critic_trainer_sparse.apply_gradients(grads_and_vars)

		save_list = tf.trainable_variables()
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
			states, actions, rewards, values, neglogprobs = zip(*data)
			values = np.concatenate((values, [0]), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			for i in reversed(range(len(data))):
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lambd * ad_t
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
		state_batch, action_batch, TD_batch, TD_sparse_batch, neglogp_batch, GAE_batch = self.computeTDandGAEAdaptive(tuples)
		if len(state_batch) < self.batch_size:
			return
		GAE_batch = (GAE_batch - GAE_batch.mean())/(GAE_batch.std() + 1e-5)

		ind = np.arange(len(state_batch))
		np.random.shuffle(ind)

		lossval_ac = 0
		lossval_c = 0
		lossval_cs = 0
		for s in range(int(len(ind)//self.batch_size)):
			selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]
			val = self.sess.run([self.actor_train_op, self.critic_train_op, self.critic_sparse_train_op,
							self.loss_actor, self.loss_critic, self.loss_critic_sparse], 
				feed_dict={
					self.state: state_batch[selectedIndex], 
					self.TD: TD_batch[selectedIndex], 
					self.TD_sparse: TD_sparse_batch[selectedIndex], 
					self.action: action_batch[selectedIndex], 
					self.old_neglogp: neglogp_batch[selectedIndex], 
					self.GAE: GAE_batch[selectedIndex],
					self.learning_rate_ph:self.learning_rate_actor
				}
			)
			lossval_ac += val[3]
			lossval_c += val[4]
			lossval_cs += val[5]
		self.lossvals = []
		self.lossvals.append(['loss actor', lossval_ac])
		self.lossvals.append(['loss critic', lossval_c])
		self.lossvals.append(['loss critic sparse', lossval_cs])

	def computeTDandGAEAdaptive(self, tuples):
		state_batch = []
		action_batch = []
		TD_batch = []
		TD_sparse_batch = []
		neglogp_batch = []
		GAE_batch = []
		self.values_sparse = 0
		self.values_dense = 0
		for data in tuples:
			size = len(data)		
			# get values
			states, actions, rewards, values, neglogprobs, times = zip(*data)

			values_dense =  np.concatenate((np.array(values)[:,0], [0]), axis=0)
			values_sparse =  np.concatenate((np.array(values)[:,1], [0]), axis=0)
			advantages_dense = np.zeros(size)
			advantages_sparse = np.zeros(size)
			ad_t_sparse = 0
			ad_t_dense = 0
			for i in reversed(range(len(data))):
				delta_dense = rewards[i][0] + values_dense[i+1] * self.gamma - values_dense[i]
				ad_t_dense = delta_dense + self.gamma * self.lambd * ad_t_dense
				advantages_dense[i] = ad_t_dense

				# if i != len(data)-1 and times[i] > times[i+1]:
				# 	delta_sparse = rewards[i][1] - values_sparse[i]
				# 	ad_t_sparse = delta_sparse
				# else:
				delta_sparse = rewards[i][1] + values_sparse[i+1] * self.gamma_sparse - values_sparse[i]
				ad_t_sparse = delta_sparse + self.gamma_sparse * self.lambd * ad_t_sparse

				advantages_sparse[i] = ad_t_sparse

			TD = values_dense[:size] + advantages_dense
			TD_sparse = values_sparse[:size] + advantages_sparse
			self.values_sparse += np.mean(values_sparse)
			self.values_dense += np.mean(values_dense)
			for i in range(size):
				state_batch.append(states[i])
				action_batch.append(actions[i])
				TD_batch.append(TD[i])
				TD_sparse_batch.append(TD_sparse[i])
				neglogp_batch.append(neglogprobs[i])
				GAE_batch.append(advantages_dense[i]+advantages_sparse[i])
		self.values_sparse /= len(tuples)
		self.values_dense /= len(tuples)
		return np.array(state_batch), np.array(action_batch), np.array(TD_batch), np.array(TD_sparse_batch), np.array(neglogp_batch), np.array(GAE_batch)

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
		if self.adaptive:
			print_list.append('avg sparse value : {:.3f}'.format(self.values_sparse))
			print_list.append('avg dense value : {:.3f}'.format(self.values_dense))
		print_list.append('===============================================================')
		for s in print_list:
			print(s)


	def train(self, num_iteration):
		epi_info_iter = []
		for it in range(num_iteration):
			if self.adaptive and it % 5 == 0:	
				self.optimizeReference(100)

			for i in range(self.num_slaves):
				self.env.reset(i)
			states = self.env.getStates()
			local_step = 0
			last_print = 0
			
			epi_info = [[] for _ in range(self.num_slaves)]	

			while True:
				# set action
				actions, neglogprobs = self.actor.getAction(states)
				if not self.adaptive:
					values = self.critic.getValue(states)
				else:
					values = [self.critic.getValue(states), self.critic_sparse.getValue(states)]
					values = np.array(values).transpose()				
				rewards, dones, times = self.env.step(actions)
				for j in range(self.num_slaves):
					if not self.env.getTerminated(j):
						if not self.adaptive and rewards[j] is not None:
							epi_info[j].append([states[j], actions[j], rewards[j], values[j], neglogprobs[j]])
							local_step += 1
						if self.adaptive and rewards[j][0] is not None:
							epi_info[j].append([states[j], actions[j], rewards[j], values[j], neglogprobs[j], times[j]])
							local_step += 1
						if dones[j]:
							if len(epi_info[j]) != 0:
								epi_info_iter.append(deepcopy(epi_info[j]))
							
							if local_step < self.steps_per_iteration:
								epi_info[j] = []
								self.env.reset(j)
							else:
								self.env.setTerminated(j)

				if local_step >= self.steps_per_iteration:
					if self.env.getAllTerminated():
						print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration),end='\r')
						break
				if last_print + 100 < local_step: 
					print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration),end='\r')
					last_print = local_step

				states = self.env.getStates()
			print('')

			if it % 5 == 4:	
		#	if 1:		
				if self.adaptive:
					self.updateAdaptive(epi_info_iter)
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

	def optimizeReference(self, num_max_iteration):

		self.env.sim_env.OptimizationStart()
		print('Optimization start')

		for it in range(num_max_iteration):
			self.env.sim_env.GenerateRandomTrajectory()

			for i in range(self.num_slaves):
				self.env.reset(i)
			states = self.env.getStates()
			
			while True:
				# set action
				actions = self.actor.getMeanAction(states)
				rewards, dones, times  = self.env.step(actions, False)
				for j in range(self.num_slaves):
					if not self.env.getTerminated(j):
						if dones[j]:
							self.env.setTerminated(j)

				if self.env.getAllTerminated():
					break

				states = self.env.getStates()

			print('Optimization: iter {}'.format(it+1),end='\r')
			t = self.env.sim_env.Optimize()
			if t:
				break

		print('')
		print('Optimization done')

	def run(self, state):
		state = np.reshape(state, (1, self.num_state))
		state = self.RMS.apply(state)
		#action, _ = self.actor.getAction(state)
		action = self.actor.getMeanAction(state)

		return action

	def eval(self):
		pass


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ntimesteps", type=int, default=1000000)
	parser.add_argument("--ref", type=str, default="")
	parser.add_argument("--test_name", type=str, default="")
	parser.add_argument("--pretrain", type=str, default="")
	parser.add_argument("--evaluation", type=bool, default=False)
	parser.add_argument("--nslaves", type=int, default=4)
	parser.add_argument("--adaptive", dest='adaptive', action='store_true')
	parser.add_argument("--save", type=bool, default=True)
	parser.add_argument("--no-plot", dest='plot', action='store_false')
	parser.set_defaults(plot=True)
	parser.set_defaults(adaptive=False)

	args = parser.parse_args()

	directory = None
	if args.save:
		if not os.path.exists("./output/"):
			os.mkdir("./output/")

		directory = "./output/" + args.test_name + "/"
		if not os.path.exists(directory):
			os.mkdir(directory)

	if args.pretrain != "":
		env = Monitor(ref=args.ref, num_slaves=args.nslaves, load=True, directory=directory, plot=args.plot, adaptive=args.adaptive)
	else:
		env = Monitor(ref=args.ref, num_slaves=args.nslaves, directory=directory, plot=args.plot, adaptive=args.adaptive)

	ppo = PPO()
	ppo.initTrain(env=env, name=args.test_name, directory=directory, pretrain=args.pretrain, evaluation=args.evaluation, 
		adaptive=args.adaptive)
	ppo.train(args.ntimesteps)
