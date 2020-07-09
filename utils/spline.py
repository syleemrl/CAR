import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from IPython import embed
from scipy.signal import butter, lfilter, filtfilt

def B(idx, t):
	if idx == 0:
		return 1.0 / 6 * pow(1 - t, 3)
	elif idx == 1:
		return 1.0 / 6 * (3 * pow(t, 3) - 6 * pow(t, 2) + 4)
	elif idx == 2:
		return 1.0 / 6 * (-3 * pow(t, 3) + 3 * pow(t, 2) + 3 * t + 1)
	else:
		return 1.0 / 6 * pow(t, 3)

def spline_to_motion(motion_size, cps, knots, idx):
	cps_size = len(knots)
	motion = []
	time = [i for i in range(motion_size)]
	for i in range(motion_size):
		b_idx = cps_size - 1
		for j in range(cps_size):
			if knots[j] > i:
				b_idx = j-1
				break
		p = 0
		knot_interval = (knots[(b_idx + 1) % cps_size] - knots[b_idx] + motion_size) % motion_size
		t = (i - knots[b_idx]) / knot_interval
		for j in range(-1, 3):
			cp_idx = (b_idx + j + cps_size) % cps_size
			p += B(j+1, t) * cps[cp_idx][idx]

		motion.append(p)
	return time, motion

def plot(filename):
	data = open(filename)
	
	knots = []
	cps_size = int(data.readline())
	for i in range(cps_size):
		l = data.readline()
		l = float(l.split()[0])
		knots.append(l)

	cps = []
	for i in range(cps_size):
		l = data.readline()
		l = [float(t) for t in l.split()]		
		cps.append(l)

	motion_size = int(data.readline())
	motion = []
	time = []
	for i in range(motion_size):
		l = data.readline()
		l = float(l.split()[0])
		time.append(l)
		l = data.readline()
		l = [float(t) for t in l.split()]		
		motion.append(l)
	data.close()
	motion = np.array(motion)
	motion_j = motion[:, 25]

	time_s, motion_j_s = spline_to_motion(motion_size, cps, knots, 25)
	plt.plot(time, motion_j)
	plt.plot(time_s, motion_j_s, 'r')
	plt.show()
	plt.savefig(filename+'.png')

if __name__=="__main__":
	plot(sys.argv[1])