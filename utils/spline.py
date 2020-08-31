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
def read_data(filename):
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
	return cps_size, knots, cps, motion_size, motion, time
def plot(filename, filename2=""):
	cps_size, knots, cps, motion_size, motion, time = read_data(filename)
	# motion_j = motion[:, 23]
	# time_s, motion_j_s = spline_to_motion(motion_size, cps, knots,23 )
	# plt.plot(time, motion_j)
	# plt.plot(time_s, motion_j_s, 'r')
	# plt.show()

	key = 3
	if filename2 != "":
		_, _, cps2, _, motion2, _ = read_data(filename2)

	cps_d = []
	for i in range(len(knots)):
		cps_d.append(np.array(cps2[i]) - np.array(cps[i]))

	idxs = []
	for i in range(len(cps_d[0])):
		flag = False
		for j in range(len(knots)):
			if cps_d[j][i] > 0.1:
				flag = True
		if flag:
			idxs.append(i)

	motion_j = motion[:, key]
	motion_j2 = motion2[:, key]

	time_s, motion_j_s = spline_to_motion(motion_size, cps, knots, key)
	time_s2, motion_j_s2 = spline_to_motion(len(motion_j2), cps2, knots, key)

	plt.figure(figsize=(12,5))
	plt.suptitle("joint"+str(key))

	plt.subplot(1, 2, 1)
	plt.gca().set_title("original")
	plt.plot(time, motion_j)
	plt.plot(time_s, motion_j_s, 'r')


	plt.subplot(1, 2, 2)
	plt.gca().set_title("cmp")
	plt.plot(time_s2, motion_j2)
	plt.plot(time_s2, motion_j_s2, 'r')

	plt.show()
	plt.cla()

def plot_solo(filename):
	key = 36
	cps_size, knots, cps, motion_size, motion, time = read_data(filename)
	motion_j = motion[:, key]
	time_s, motion_j_s = spline_to_motion(motion_size, cps, knots, key)

	plt.plot(time, motion_j)
	plt.plot(time_s, motion_j_s, 'r')
	plt.show()

if __name__=="__main__":
	if len(sys.argv) == 2:
		plot_solo(sys.argv[1])
	else:
		plot(sys.argv[1], sys.argv[2])
