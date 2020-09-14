import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from IPython import embed
from scipy.signal import butter, lfilter, filtfilt

def plot(filename):
	data = open(filename)
	
	cps = []

	while True:
		l = data.readline()
		if not l: 
			break
		l = [float(t) for t in l.split()]		
		cps.append(np.mean(l))

	print(cps)
	plt.plot(cps, 'r')
	plt.show()

if __name__=="__main__":
	plot(sys.argv[1])