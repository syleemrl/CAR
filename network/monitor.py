from env import Env
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils import RunningMeanStd
from IPython import embed

class Monitor(object):
	def __init__(self, motion, num_slaves, directory, load=False, plot=True, verbose=True):
		self.env = Env(motion, num_slaves)
		self.num_slaves = self.env.num_slaves
		self.motion = self.env.motion
		self.sim_env = self.env.sim_env
		
		self.num_state = self.env.num_state
		self.num_action = self.env.num_action
		self.RMS = RunningMeanStd(shape=(self.num_state))	
		self.verbose = verbose
		self.plot = plot
		self.directory = directory

		#load RMS
		if load:
			self.RMS.load(self.directory+'rms')

		self.start_time = time.time()		
		self.num_evaluation = 0
		self.num_episodes = 0
		self.num_transitions = 0
		self.total_rewards = []
		self.max_episode_length = 0
		self.total_rewards_by_parts = np.array([[]]*6)
		self.transition_per_episodes = []
		self.num_nan_per_iteration = 0
		self.num_episodes_per_iteration = 0
		self.num_transitions_per_iteration = 0
		self.rewards_per_iteration = 0
		self.rewards_by_part_per_iteration = []

		self.terminated = [False]*self.num_slaves
		self.states = [0]*self.num_slaves

		self.skel_len = 1
		# self.w = 1.05
		# for _ in range(8):
		#  	self.skel_len *= w
		#  	self.sim_env.DeformCharacter(w)
		# print(self.skel_len)	 	
		if self.plot:
			plt.ion()

	def getStates(self):
		return np.array(self.states).astype('float32') 

	def setTerminated(self, idx):
		self.terminated[idx] = True
	
	def getTerminated(self, idx):
		return self.terminated[idx]
	
	def getAllTerminated(self):
		for i in range(self.num_slaves):
			if not self.terminated[i]:
				return False
		return True 
	
	def reset(self, i):
		self.env.reset(i)
		state = np.array([self.sim_env.GetState(i)])
		self.states[i] = self.RMS.apply(state)[0]
		self.terminated[i] = False
	
	def step(self, actions):
		self.states, rewards, dones, times, nan_count =  self.env.step(actions)
		states_updated = self.RMS.apply(self.states[~np.array(self.terminated)])
		self.states[~np.array(self.terminated)] = states_updated
		self.num_nan_per_iteration += nan_count
		for i in range(self.num_slaves):
			if not self.terminated[i] and rewards[i][0] is not None:
				self.rewards_per_iteration += rewards[i][0]
				self.rewards_by_part_per_iteration.append(rewards[i])
				
				self.num_transitions_per_iteration += 1
				if dones[i]:
					self.num_episodes_per_iteration += 1
					if times[i] > self.max_episode_length:
						self.max_episode_length = times[i]
			
		rewards = [rewards[i][0] for i in range(len(rewards))]

		return rewards, dones

	def plotFig(self, y_list, title, num_fig=1, ylim=True, path=None):
		if self.plot:
			plt.figure(num_fig, clear=True, figsize=(5.5, 4))
		else:
			plt.figure(num_fig, figsize=(5.5, 4))
		plt.title(title)

		i = 0
		for y in y_list:
			plt.plot(y[0], label=y[1])
			i+= 1

		plt.legend(loc=2)
		if self.plot:
			plt.show()
			if ylim:
				plt.ylim([0,1])
			plt.pause(0.001)
		if path is not None:
			plt.savefig(path, format="png")

	def printSummary(self):
		self.num_transitions += self.num_transitions_per_iteration
		self.num_episodes += self.num_episodes_per_iteration
		self.num_evaluation += 1
		self.total_rewards.append(self.rewards_per_iteration/self.num_episodes_per_iteration)
		self.total_rewards_by_parts = np.insert(self.total_rewards_by_parts, self.total_rewards_by_parts.shape[1], 
			np.asarray(self.rewards_by_part_per_iteration).sum(axis=0)/self.num_episodes_per_iteration, axis=1)

		print_list = []
		print_list.append('===============================================================')
		print_list.append('skel length: {}'.format(self.skel_len))
		print_list.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print_list.append("Elapsed time : {:.2f}s".format(time.time() - self.start_time))
		print_list.append('Num eval : {}'.format(self.num_evaluation))
		print_list.append('total episode count : {}'.format(self.num_episodes))
		print_list.append('total transition count : {}'.format(self.num_transitions))
		t_per_e = 0
		if self.num_episodes is not 0:
			t_per_e = self.num_transitions / self.num_episodes

		print_list.append('total transition per episodes : {:.2f}'.format(t_per_e))
		print_list.append('episode count : {}'.format(self.num_episodes_per_iteration))
		print_list.append('transition count : {}'.format(self.num_transitions_per_iteration))
		
		t_per_e = 0
		if self.num_episodes_per_iteration is not 0:
			t_per_e = self.num_transitions_per_iteration / self.num_episodes_per_iteration
		self.transition_per_episodes.append(t_per_e)

		print_list.append('transition per episodes : {:.2f}'.format(t_per_e))
		print_list.append('rewards per episodes : {:.2f}'.format(self.total_rewards[-1]))
		print_list.append('max episode length : {}'.format(self.max_episode_length))

		if self.num_nan_per_iteration != 0:
			print_list.append('nan count : {}'.format(self.num_nan_per_iteration))
		print_list.append('===============================================================')

		for s in print_list:
			print(s)
		
		if self.directory is not None:
			out = open(self.directory+"results", "a")
			for s in print_list:
				out.write(s+'\n')
			out.close()

			self.RMS.save(self.directory+'rms')

		if self.plot:
			y_list = [[np.asarray(self.transition_per_episodes), 'steps'], 
						[np.asarray(self.total_rewards_by_parts[0]), 'r'], 
						[np.asarray(self.total_rewards_by_parts[1]), 'p'], 
						[np.asarray(self.total_rewards_by_parts[2]), 'v'], 
						[np.asarray(self.total_rewards_by_parts[3]), 'com'],
						[np.asarray(self.total_rewards_by_parts[4]), 'ee'],
						[np.asarray(self.total_rewards_by_parts[5]), 'srl']]

			self.plotFig(y_list, "rewards" , 1, False, path=self.directory+"result.png")

			y_list = y_list[1:]
			for i in range(len(y_list)):
				y_list[i][0] = np.array(y_list[i][0])/np.array(self.transition_per_episodes)

			self.plotFig(y_list, "rewards_per_step", 2, False, path=self.directory+"result_per_step.png")

		self.num_nan_per_iteration = 0
		self.num_episodes_per_iteration = 0
		self.num_transitions_per_iteration = 0
		self.rewards_per_iteration = 0
		self.rewards_by_part_per_iteration = []

