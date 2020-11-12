import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
import math
from IPython import embed
from scipy.signal import butter, lfilter, filtfilt
from scipy.interpolate import make_interp_spline, BSpline

def plot(filename):
	data = open(filename)
	
	time = []
	phase = []
	pair = []

	length = int(data.readline())
	for l in data.readlines():
		l = [float(l.split()[0]), float(l.split()[1]), float(l.split()[2])]
		# if l[1] > length:
		# 	break
		# time.append(l[0])
		# phase.append(l[1])
		pair.append([math.fmod(l[1], length), l[2]])

	data.close()
	pair = sorted(pair, key = lambda x: x[0])
	pair = np.array(pair)
	phase = pair[:, 0]
	dphase = pair[:, 1]

	xnew = []
	ynew = []
	for i in range(length - 1) :
		xnew.append(i)
		mean = 0
		count = 0
		for j in range(len(phase)):
			if phase[j] >= i - 0.5 and phase[j]  <= i + 0.5:
				mean += dphase[j]
				count += 1
		mean /= count
		ynew.append(mean) 

	plt.plot(xnew, ynew, color='red')
	plt.savefig(filename+'.png') 
	plt.show()

if __name__=="__main__":
	plot(sys.argv[1])