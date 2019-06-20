from network import Actor
from network import Critic
from monitor import Monitor
import argparse
import random
import numpy as np
import tensorflow as tf
import pickle
import datetime

class PPO(object):
	def __init__(self, env, name, pretrain, evaluation, 
		learning_rate_actor=2e-4, learning_rate_critic=0.001, learning_rate_decay=0.9993,
		gamma=0.99, lambd=0.95, epsilon=0.2, directory=None,
		batch_size=1024, steps_per_iteration=20000):

		self.name = name
		self.evaluation = evaluation
		self.num_slaves = num_slaves
		self.learning_rate_critic = learning_rate_critic
		self.learning_rate_actor = learning_rate_actor
		self.learning_rate_decay = learning_rate_decay
		self.epsilon = epsilon
		self.gamma = gamma
		self.lambd = lambd
		self.directory = directory
		self.steps_per_iteration = steps_per_iteration
		self.batch_size = batch_size
		self.env = env
		self.num_slaves = self.env.num_slaves
		self.key_to_idx = {'S':0, 'A':1, 'R':2, 'value':3, 'neglogprob':4, 'TD':5, 'GAE':6}
		
		#build network and optimizer
		self.state = tf.placeholder(tf.float32, shape=[None, self.env.num_state], name='state')
		self.actor = Actor(self.sess, 'Actor', self.state, self.env.num_action)
		self.critic = Critic(self.sess, 'Critic', self.state)
		self.buildOptimize()
		self.sess.run(tf.global_variables_initializer())

		# set etc
		random.seed(int(time.time()))
		np.random.seed(int(time.time()))
		tf.set_random_seed(int(time.time()))

		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = self.num_slaves
		config.inter_op_parallelism_threads = self.num_slaves
		
		save_list = tf.trainable_variables()
		self.saver = tf.train.Saver(var_list=save_list,max_to_keep=1)
		
		# load pretrained network
		if pretrain is not None:
			self.load(pretrain)

		self.start_time = time.time()
		self.sim_time = 0
		self.train_time = 0

		self.printSetting()
	
	def printSetting(self):
		
		print_list = []
		print_list.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print_list.append("test_name : {}".format(self.name))
		print_list.append("motion : {}".format(self.motion))
		print_list.append("num_slaves : {}".format(self.num_slaves))
		print_list.append("num state : {}".format(self.env.num_state))
		print_list.append("num action : {}".format(self.env.num_action))
		print_list.append("learning_rate : {}".format(self.learning_rate))
		print_list.append("gamma : {}".format(self.gamma))
		print_list.append("lambd : {}".format(self.lambd))
		print_list.append("batch_size : {}".format(self.batch_size))
		print_list.append("steps_per_iteration : {}".format(self.steps_per_iteration))
		print_list.append("clip ratio : {}".format(self.epsilon))
		print_list.append("pretrain : {}".format(self.pretrain))
		print_list.append("trajectory frame : {}".format(self.env.traj_frame))

		for s in print_list:
			print(s)

		if self.directory is not None:
			out = open(self.directory+"parameters", "w")
			for s in print_list:
				out.write(s + "\n")
			out.close()

			out = open(self.directory+"results", "w")
			out.close()

	def buildOptimize(self):
		with tf.variable_scope('Optimize'):
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
			reg_l2_actor = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'Actor')
			reg_l2_critic = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'Critic')

			loss_actor = surrogate + tf.reduce_sum(reg_l2_actor)
			loss_critic = value_loss + tf.reduce_sum(reg_l2_critic)

		actor_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
		grads, params = zip(*actor_trainer.compute_gradients(loss_actor));
		grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		self.actor_train_op = actor_trainer.apply_gradients(grads_and_vars)

		critic_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
		grads, params = zip(*critic_trainer.compute_gradients(loss_critic));
		grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		self.critic_train_op = critic_trainer.apply_gradients(grads_and_vars)

	def update(self, tuples):
		if len(tuples) < self.batch_size:
			return
		tuples = self.computeTDandGAE(tuples)

		GAE = tuples[:,key_to_idx['GAE']]
		GAE = (GAE - GAE.mean())/(GAE.std() + 1e-5)

		ind = np.arange(len(GAE))
		np.random.shuffle(ind)
		for s in range(int(len(ind)//self.batch_size)):
			selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]
			batch = tuples[selectedIndex]

			self.sess.run([self.actor_train_op, self.critic_train_op], 
				feed_dict={
					self.state:batch[:,key_to_idx['S']], 
					self.TD:batch[:,key_to_idx['TD']], 
					self.action:batch[:,key_to_idx['A']], 
					self.old_neglogprobs:batch[:,key_to_idx['neglogprob']], 
					self.GAE:GAE[selectedIndex],
					self.learning_rate_ph:self.learning_rate_actor
				}
			)
				
	def computeTDandGAE(self, tuples):
		result = []
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
				result.append([states[i], actions[i], rewards[i], values[i], neglogprobs[i], TD[i], advantages[i]])
		return np.array(result)

	def save(self):
		self.saver.save(self.sess, self.directory + "_network", global_step = 0)

	def load(self, path):
		self.saver.restore(self.sess, path)

	def train(self, num_iteration):

		for it in range(num_iteration):
			for i in range(self.num_slaves):
				self.env.reset(i)

			states = self.env.updateAndGetStates()

			actions = [None]*self.num_slaves
			rewards = [None]*self.num_slaves
			episodes = [None]*self.num_slaves
				
			local_step = 0
			last_print = 0
		
			epi_info_iter = []
			epi_info = [[] for _ in range(self.num_slaves)]	
			while True:
				# set action
				actions, neglogprobs = self.actor.getAction(states)
				values = self.critic.getValue(states)
				next_states, rewards, dones = self.env.step(actions)

				for j in range(self.num_slaves):
					if not self.getTerminated(j):
						if rewards[j] is not None:
							epi_info[j].append([states[j], actions[j], rewards[j], values[j], neglogprobs[j]])
							local_step += 1

						if dones[j]:
							epi_info_iter.append(deepcopy(epi_info[j]))

							if local_step < self.steps_per_iteration:
								epi_info[j] = []
								self.env.reset(j)
							else:
								self.env.setTerminated(j)

				if local_step >= self.steps_per_iteration:
					if all(t is True for t in terminated):
						print('{}/{} : {}/{}'.format(it+1, num_iteration, local_step, self.steps_per_iteration),end='\r')
						break
				if last_print + 100 < local_step: 
					print('{}/{} : {}/{}'.format(it+1, num_iteration, local_step, self.steps_per_iteration),end='\r')
					last_print = local_step

				# update states	
				states = next_states			

			print('')
			self.update(epi_info_iter) 

		if self.learning_rate > 1e-5:
			self.learning_rate = self.learning_rate * self.learning_rate_decay

		if self.directory is not None:
			self.save()
		self.env.printSummary()

	def eval(self):
		pass

	def run(self):
		pass

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ntimesteps", type=int, default = 1000000)
	parser.add_argument("--motion", type=str, default=None)
	parser.add_argument("--name", type=str, default=None)
	parser.add_argument("--pretrain", type=str, default=None)
	parser.add_argument("--evaluation", type=bool, default=False)
	parser.add_argument("--nslaves", type=int, default=4)
	parser.add_argument("--save", type=bool, default=True)
	args = parser.parse_args()

	directory = None
	if args.save:
		if not os.path.exists("./output/"):
			os.mkdir("./output/")
		directory = "./output/" + name + /
		if not os.path.exists(directory):
			os.mkdir(directory)
	
	env = Monitor(args.motion, args.nslaves, directory=directory)
	ppo = PPO(env=env, name=args.name, directory=directory, pretrain=args.pretrain, evaluation=args.evaluation)
	ppo.train(args.ntimesteps)