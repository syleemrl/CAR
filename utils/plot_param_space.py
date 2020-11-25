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

def plot(filename):
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
			if value[count] > 1.5:
				v.append(1.5)
			else:
				v.append(value[count])
			d.append(density[count])
			f.append(fitness[count])
			count += 1

		V.append(v)
		F.append(f)
		D.append(d)

	V = np.array(V)
	F = np.array(F)
	D = np.array(D)

	fig = plt.figure('value')
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, V, cmap=cm.coolwarm,
                linewidth=0, antialiased=True)

	fig = plt.figure('density')
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, D, cmap=cm.coolwarm,
                linewidth=0, antialiased=True)


	fig = plt.figure('fitness')
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, F, cmap=cm.coolwarm,
                linewidth=0, antialiased=True)

	plt.show()

if __name__=="__main__":
	plot(sys.argv[1])
