import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
import math
from IPython import embed
from scipy.signal import butter, lfilter, filtfilt
from scipy.interpolate import make_interp_spline, BSpline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def read_data(filename):
	data = open(filename)
	
	xy = []
	density = []
	value = []
	fitness = []

	for l in data.readlines():
		l = [float(t) for t in l.split()]
		xy.append([l[0], l[1]])
		value.append(l[2])
		density.append(l[3])
		fitness.append(l[4])
		
	x = np.linspace(0.05, 0.95, 19)
	y = np.linspace(0.05, 0.95, 19)
	X, Y = np.meshgrid(x, y)
	XY = np.dstack((X, Y))
	V = []
	F = []
	D = []

	count = 0
	for pair in XY:
		v = []
		d = []
		f = []
		for i in range(19):
			if density[count] > 0.5:
				v.append(value[count])
			else:
			 	v.append(0)
			d.append(density[count])
			f.append(fitness[count])
			count += 1

		V.append(v)
		F.append(f)
		D.append(d)

	V = np.array(V)
	F = np.array(F)
	D = np.array(D)

	return X, Y, V, D, F
def plot():

	Xs = []
	Ys = []
	Vs = []
	Ds = []
	Fs = []
	filelist = []
	for i in range(6):
		filelist.append('param_summary'+str(10 * (i+1)))
	for i in range(len(filelist)):
		X, Y, V, D, F = read_data('../network/output/punch_ue_2d_test/'+filelist[i])
		Xs.append(X)
		Ys.append(Y)
		Vs.append(V)
		Ds.append(D)
		Fs.append(F)

	fig = plt.figure('value', figsize=(30,10))
	col_size = int((len(filelist) + 1) / 2)
	for i in range(len(filelist)):
		ax = fig.add_subplot(2, col_size, i+1, projection='3d')
		surf = ax.plot_surface(Xs[i], Ys[i], Vs[i], cmap=cm.coolwarm,
	                linewidth=0, antialiased=True)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_title(i)
		ax.azim = 0
		ax.elev = 90
	fig.tight_layout()

	fig = plt.figure('density', figsize=(30,10))

	for i in range(len(filelist)):
		ax = fig.add_subplot(2, col_size, i+1, projection='3d')
		surf = ax.plot_surface(Xs[i], Ys[i], Ds[i], cmap=cm.coolwarm, vmin=0, vmax=1.2,
	                linewidth=0, antialiased=True)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_title(i)
		ax.azim = 0
		ax.elev = 90
	fig.tight_layout()

	# fig = plt.figure('fitness')
	# ax = fig.gca(projection='3d')
	# surf = ax.plot_surface(X, Y, F, cmap=cm.coolwarm,
 #                linewidth=0, antialiased=True)

	plt.show()

if __name__=="__main__":
	plot()