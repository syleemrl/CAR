import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math

def pca(data):
	x = StandardScaler().fit_transform(data)
	pca = PCA(n_components = 2)
	principalComponents = pca.fit_transform(x)
	return principalComponents

def plot(file):
	data = []
	f = open(file, 'r')
	for line in f.readlines():
		d = line.strip().split()
		d = [float(f) for f in d]
		data.append(d)
	f.close()
	data = np.array(data)
	angles = data[:,:-1]
	target = data[:,-1]

	fig = plt.figure()	
	size = int(len(angles[0]) / 3)
	row = math.ceil(size / 3)
	count = 1
	for i in range(0, len(angles[0]), 3):
		joint_angles = angles[:,i:i+3]
		features = pca(joint_angles)
		x = features[:,0]
		y = features[:,1]

		ax = fig.add_subplot(row, 3, count, projection='3d')
		ax.scatter(x, y, target)
		count += 1
	
	plt.show()

if __name__=="__main__":
	plot(sys.argv[1])