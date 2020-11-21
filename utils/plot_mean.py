import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from IPython import embed
from sklearn.neighbors import KNeighborsRegressor
from scipy.signal import butter, lfilter, filtfilt
def gaussian(x, y):
	return 1 / np.sqrt(2*np.pi) * np.exp(- (x-y)**2 / 2)

def plot(filename):
	data = open(filename)
	
	initial = []
	progress = []
	while True:
		l = data.readline()
		if not l: 
			break
		l = [float(t) for t in l.split('/')]
		initial.append(l[0])		

		l = data.readline()
		l = [float(t) for t in l.split()]		
		progress.append(l[0])

	# initial = [1.0732892685022162, 1.0136029012863919, 1.0428629088952261, 0.9839837313528164, 0.983998239328098, 1.0884403356023604, 0.9804834736767556, 0.9488263872299297, 1.063532624778015, 0.9225140991865576, 1.0356094555818198, 0.9655642111528151, 1.0584820172469722, 0.9665887558068448, 0.9896257844939073, 1.088250637832992, 1.0896717454109301, 1.0286747291665903, 0.9624228682414007, 0.9819473264399546]
	# progress = [-0.003198407413745441, 0.009515485612859642, -0.009340054257909847, -0.01385610008519167, 0.013111487047577142, 0.01781728526527626, 0.032942001519474906, -0.003404240176300366, -0.00417988870466357, 0.0015719353749719112, 0.0098566007070795, 0.0020872027835618923, 0.009968379093298063, -0.007444366235816902, -0.022024968733875716, 0.004721402875115466, 0.017324876387798005, -0.00013923890832723274, -0.012738646781706664, 0.008467414671113005]
	initial = np.array(initial).reshape(-1, 1)
	regressor = KNeighborsRegressor(n_neighbors=20, weights="distance", metric='minkowski')
	regressor.fit(initial, progress)

	x = np.linspace(0.8, 1.2, num=40)
	x_ = np.array(x).reshape(-1, 1)
	y = regressor.predict(x_)
	# y = []
	# for j in range(len(x)):
	# 	d = 0
	# 	div = 0
	# 	for i in range(len(initial)):
	# 		g = gaussian(initial[i], x[j])
	# 		div += g
	# 		d += g * progress[i]
	# 	embed()
	# 	y.append(d / div)

	plt.plot(x, y)
#	plt.scatter(initial, progress)
	plt.show()

if __name__=="__main__":
	plot(sys.argv[1])