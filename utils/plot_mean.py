import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from IPython import embed
from sklearn.neighbors import KNeighborsRegressor
#import tensorflow as tf
import random
import time
# activ = tf.nn.elu
# kernel_initialize_func = tf.contrib.layers.xavier_initializer()
# regression_layer_size = 64
# l2_regularizer_scale = 0.0

# class RegressionNet(object):
# 	def __init__(self, sess, input, output):
# 		self.sess = sess
# 		self.value = self.createNetwork(input, output, False)

# 		self.input = input
# 	def createNetwork(self, input, output, reuse):	
# 		L1 = tf.layers.dense(input, regression_layer_size, activation=activ,name='L1',
# 	        kernel_initializer=kernel_initialize_func,
# 	        kernel_regularizer=None
# 		)

# 		L2 = tf.layers.dense(L1, regression_layer_size, activation=activ,name='L2',
# 	        kernel_initializer=kernel_initialize_func,
# 	        kernel_regularizer=None
# 		)

# 		out = tf.layers.dense(L2, output, name='out',
# 	        kernel_initializer=kernel_initialize_func,
# 	        kernel_regularizer=None
# 		)

# 		return out
# 	def getValue(self, states):
# 		return self.sess.run(self.value, feed_dict={self.input :states})

# 	def getVariable(self, trainable_only=False):
# 		if trainable_only:
# 			return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
# 		else:
# 			return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
# class Regression(object):
# 	def __init__(self,learning_rate=2e-4,learning_rate_decay=0.9993, input_dim=1, output_dim=1):
# 		random.seed(int(time.time()))
# 		np.random.seed(int(time.time()))
# 		tf.set_random_seed(int(time.time()))
	
# 		self.learning_rate_decay = learning_rate_decay
# 		self.learning_rate = learning_rate	

# 		self.input_dim = input_dim
# 		self.output_dim = output_dim

# 		self.batch_size = 128

# 		config = tf.ConfigProto()
# 		self.sess = tf.Session(config=config)
# 		self.buildOptimize()
# 		self.regression_x = np.empty(shape=[0, self.input_dim])
# 		self.regression_y = np.empty(shape=[0, self.output_dim])

# 	def buildOptimize(self):

# 		self.input = tf.placeholder(tf.float32, shape=[None, self.input_dim])
# 		self.output = tf.placeholder(tf.float32, shape=[None, self.output_dim])

# 		self.regression = RegressionNet(self.sess, self.input, self.output_dim)
# 		self.loss_regression = tf.reduce_mean(tf.square(self.regression.value - self.output))
# 		regression_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
# 		self.regression_train_op = regression_trainer.minimize(self.loss_regression)

# 		self.sess.run(tf.global_variables_initializer())

# 	def setTrainingData(self, x, y):

# 		self.regression_x = np.array(x)
# 		self.regression_y = np.array(y)

# 	def train(self):
# 		self.lossvals = []
# 		lossval_reg = 0
# 		lossval_reg_prev = 1e8
# 		epsilon_count = 0
# 		n_iteration = 0
# 		while epsilon_count < 5:
# 			n_iteration += 1
# 			lossval_reg_prev = lossval_reg
# 			lossval_reg = 0
# 		# for it in range(self.steps_per_iteration * n):
# 			if int(len(self.regression_x) // self.batch_size) == 0:

# 				val = self.sess.run([self.regression_train_op, self.loss_regression], 
# 						feed_dict={
# 							self.input: self.regression_x, 
# 							self.output: self.regression_y, 
# 						}
# 					)
# 				lossval_reg += val[1]

# 			else:
# 				ind = np.arange(len(self.regression_x))
# 				np.random.shuffle(ind)

# 				for s in range(int(len(ind)//self.batch_size)):
# 					selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]

# 					val = self.sess.run([self.regression_train_op, self.loss_regression], 
# 						feed_dict={
# 							self.input: self.regression_x[selectedIndex], 
# 							self.output: self.regression_y[selectedIndex], 
# 						}
# 					)
# 					lossval_reg += val[1]
# 			if abs(lossval_reg_prev - lossval_reg) < 1e-5:
# 				epsilon_count += 1

def plot(filename):
	data = open(filename)
	
	initial = []
	progress = []
	while True:
		l = data.readline()
		if not l: 
			break
		l = [float(t) for t in l.split(', ')]
		if l[0] > 0.7 and abs(l[1]) < 0.1:
			initial.append(l[0])			
			progress.append(l[1])

	initial = np.array(initial).reshape(-1, 1)
	# progress = np.array(progress).reshape(-1, 1)

	# regressor = Regression()
	# regressor.setTrainingData(initial, progress)
	# regressor.train()
	# initial = np.array(initial).reshape(-1, 1)
	regressor = KNeighborsRegressor(n_neighbors=10, weights="uniform")
	regressor.fit(initial, progress)

	x = np.linspace(0.6, 1.2, num=60)
	x_ = np.array(x).reshape(-1, 1)
#	y = regressor.regression.getValue(x_)
	y = regressor.predict(x_)

	plt.plot(x, y, color='r')
	plt.scatter(initial, progress)
	plt.savefig(filename+'.png') 
	plt.show()

if __name__=="__main__":
	plot(sys.argv[1])