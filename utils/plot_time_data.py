import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
import math
from IPython import embed
from scipy.signal import butter, lfilter, filtfilt
from scipy.interpolate import make_interp_spline, BSpline

def plot():
	data = open('../network/output/jump2_mxm_allc2_x/0.5_time2')
	
	time = []
	phase = []

	count = 0
	for l in data.readlines():
		p = float(l.split()[0])
		if p > 66:
			break
		phase.append(p)
		time.append(count)
		count += 1
	standard = [1] * len(time)
	plt.rc('axes', labelsize=12)

	plt.xlabel(r'$\phi$')
	plt.ylabel(r'$\Delta\phi$')
	plt.xlim([time[0], time[-1]])
	plt.ylim([0, 3])
	plt.plot(time, phase, color='blue')
	plt.axvline(x=22, color='r', linestyle='--', linewidth=1)
	plt.axvline(x=38, color='r', linestyle='--', linewidth=1)
	# plt.text(22, 1.19,'take-off', horizontalalignment='center', color='r', fontsize=10)
	# plt.text(38, 1.19, 'landing', horizontalalignment='center', color='r', fontsize=10)

	plt.show()

if __name__=="__main__":
	plot()