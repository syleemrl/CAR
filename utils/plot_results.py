import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from scipy.signal import butter, lfilter, filtfilt
joint = ['Spine', 'Neck', 'Head', 'ArmL', 'ForeArmL', 'HandL', 'ArmR', 'ForeArmR', 'HandR', 'FemurL',
 'TibiaL', 'FootL', 'FootEndL', 'FemurR', 'TibiaR', 'FootR', 'FootEndR']
grf_joint = ['FootR', 'FootL', 'FootEndR', 'FootEndL']
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def read_data_joint(data):
	result = []
	size = int(data.readline())
	for i in range(size):
		l = data.readline()
		l = [float(t) for t in l.split()]
		l = l[6:]
		result.append(l)

	x_list = [[] for _ in range(int(len(result[0]) / 3))]
	y_list = [[] for _ in range(int(len(result[0]) / 3))]
	z_list = [[] for _ in range(int(len(result[0]) / 3))]


	for j in range(len(result)):
		for i in range(len(result[0])):
			if i % 3 == 0:
				x_list[int(i / 3)].append(result[j][i])
			elif i % 3 == 1:
				y_list[int(i / 3)].append(result[j][i])
			else:
				z_list[int(i / 3)].append(result[j][i])

	return x_list, y_list, z_list

def read_data_double(data):
	result = []
	size = int(data.readline())
	for i in range(size):
		l = data.readline()
		l = float(l.split()[0])
		result.append(l)
	return result

def plot_data_joint(data, directory, filename):
	plt.figure(figsize=(15,12))
	plt.subplots_adjust(hspace = 0.4, wspace = 0.15)

	cnt = 1
	for d in data:
		plt.subplot(6, 3, cnt)
		plt.gca().set_title(joint[cnt-1])
		plt.plot(d, color='blue')
		cnt += 1
	plt.savefig(directory+'_'+filename+'.png') 

def plot_data_double(data, directory, filename):
	plt.figure(figsize=(20,10))
	plt.plot(data, color='red')
	# plt.savefig(directory+'_'+filename+'.png') 

def plot(filename):
	data = open(filename)
	# if not os.path.exists('./result_'+filename):
	# 	os.mkdir('result_'+filename)
	# directory = './result_'+filename+'/'+filename
	
#	x1, y1, z1 = read_data_joint(data)
	w = read_data_double(data)
#	x2, y2, z2 = read_data_joint(data)
	data.close()

	# time_dt = [t-20 for t in time]
	plot_data_double(w, "", 'w')
#	plot_data_joint(x1, directory, 'tqx')
#	plot_data_joint(y1, directory, 'tqy')
#	plot_data_joint(z1, directory, 'tqz')

	plt.show()

if __name__=="__main__":
	plot(sys.argv[1])