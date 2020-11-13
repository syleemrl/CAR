from network import RegressionNet
import random
import numpy as np
import tensorflow as tf
import pickle
import os
import time
import sys
from IPython import embed
from copy import deepcopy
from tensorflow.python import pywrap_tensorflow

import types

if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
	tf.contrib._warning = None

class Regression(object):
	def __init__(self, learning_rate=2e-4,learning_rate_decay=0.9993):
		random.seed(int(time.time()))
		np.random.seed(int(time.time()))
		tf.set_random_seed(int(time.time()))
	
		self.learning_rate_decay = learning_rate_decay
		self.learning_rate = learning_rate	

	def initRun(self, directory, num_input, num_output, postfix=""):
		self.directory = directory

		self.num_input = num_input
		self.num_output = num_output

		config = tf.ConfigProto()
		self.sess = tf.Session(config=config)
		#build network and optimizer
		self.name = directory.split("/")[-2] + postfix
		self.postfix = postfix
		self.buildOptimize(self.name)

		self.load()

	def initTrain(self, directory, num_input, num_output, postfix="",
		batch_size=128, steps_per_iteration=1):
		name = directory.split("/")[-2]
		self.name = name + postfix
		self.postfix = postfix

		self.directory = directory
		self.steps_per_iteration = steps_per_iteration
		self.batch_size = batch_size

		self.num_input = num_input
		self.num_output = num_output

		config = tf.ConfigProto()
		self.sess = tf.Session(config=config)

		#build network and optimizer
		self.buildOptimize(self.name)
		self.regression_x = np.empty(shape=[0, num_input])
		self.regression_y = np.empty(shape=[0, num_output])

		self.load()

		print("init regression network done")

	def buildOptimize(self, name):

		self.input = tf.placeholder(tf.float32, shape=[None, self.num_input], name=name+'_input')
		self.output = tf.placeholder(tf.float32, shape=[None, self.num_output], name=name+'_output')

		self.regression = RegressionNet(self.sess, name, self.input, self.num_output)
		self.loss_regression = tf.reduce_mean(tf.square(self.regression.value - self.output))
		regression_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.regression_train_op = regression_trainer.minimize(self.loss_regression)

		var_list = tf.trainable_variables()
		save_list = []
		for v in var_list:
			if name in v.name:
				save_list.append(v)

		self.saver = tf.train.Saver(var_list=save_list, max_to_keep=1)
		self.save_list = save_list

		self.sess.run(tf.global_variables_initializer())


	def setRegressionData(self, tuples):
		self.regression_x = tuples[0]
		self.regression_y = tuples[1]

	def appendRegressionData(self, tuples):
		if len(tuples[0]) == 0:
			return
		self.regression_x = np.concatenate((self.regression_x, tuples[0]), axis=0)
		self.regression_y = np.concatenate((self.regression_y, tuples[1]), axis=0)

	def save(self):
		self.saver.save(self.sess, self.directory + "reg_network" + self.postfix, global_step = 0)

	def load(self):
		path = self.directory + "reg_network" + self.postfix + "-0"
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
		
		try:
			saved_variables, saved_values = get_tensors_in_checkpoint_file(path)
			saved_dict = {n : v for n, v in zip(saved_variables, saved_values)}
			restore_op = []

			for v in self.save_list:
				# if v.name[0:11]+v.name[14:-2] in saved_dict:
				# 	saved_v = saved_dict[v.name[0:11]+v.name[14:-2]]

				if v.name[:-2] in saved_dict:
					saved_v = saved_dict[v.name[:-2]]
					if v.shape == saved_v.shape:
						print("Restore {}".format(v.name[:-2]))
						restore_op.append(v.assign(saved_v))

			restore_op = tf.group(*restore_op)
			self.sess.run(restore_op)
		except:
			print("Nothing to load")

	def printNetworkSummary(self):
		print_list = []
		print_list.append('===============================================================')
		for v in self.lossvals:
			print_list.append('{}: {:.3f}'.format(v[0], v[1]))
		print_list.append('===============================================================')
		for s in print_list:
			print(s)


	def train(self, max_iter):
		self.lossvals = []
		lossval_reg = 0
		lossval_reg_prev = 1e8
		epsilon_count = 0
		n_iteration = 0
		while epsilon_count < 2:
			n_iteration += 1
			lossval_reg_prev = lossval_reg
			lossval_reg = 0
		# for it in range(self.steps_per_iteration * n):
			if int(len(self.regression_x) // self.batch_size) == 0:

				val = self.sess.run([self.regression_train_op, self.loss_regression], 
						feed_dict={
							self.input: self.regression_x, 
							self.output: self.regression_y, 
						}
					)
				lossval_reg += val[1]

			else:
				ind = np.arange(len(self.regression_x))
				np.random.shuffle(ind)

				for s in range(int(len(ind)//self.batch_size)):
					selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]

					val = self.sess.run([self.regression_train_op, self.loss_regression], 
						feed_dict={
							self.input: self.regression_x[selectedIndex], 
							self.output: self.regression_y[selectedIndex], 
						}
					)
					lossval_reg += val[1]
			if n_iteration % 10 == 0:
				print(n_iteration, lossval_reg)
			if abs(lossval_reg_prev - lossval_reg) < 1e-4:
				epsilon_count += 1
			if n_iteration > max_iter:
				break
		self.lossvals.append(['num iteration', n_iteration])
		self.lossvals.append(['loss regression', lossval_reg])

		self.printNetworkSummary()
		self.save()

	def run(self, input):
		input = np.reshape(input, (-1, self.num_input))
		output = self.regression.getValue(input)
		return output

	def runBatch(self, input_li):
		output_li = []

		for input in input_li:
			input = np.reshape(input, (-1, self.num_input))
			output = self.regression.getValue(input)
			output_li.append(output)

		return output_li