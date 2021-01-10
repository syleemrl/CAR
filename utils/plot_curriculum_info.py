import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from IPython import embed
import random
import time
def read_vratio_type1(filename):
	data = open(filename)

	epi = []
	mode = []
	value = []
	rate = []
	while True:
		l = data.readline()
		if not l: 
			break
		l = l.split(':')
		epi.append(float(l[1])*3)
		mode.append(int(l[2]))
		l = data.readline()
		l = l.split(':')
		value.append(float(l[0]))
		rate.append(float(l[1]))

	data.close()
	return epi, mode, value, rate

def plot():
	epi, mode, value, rate = read_vratio_type1('curriculum_info')
	for i in reversed(range(len(value))):	
		if i != 0:
			value[i] = value[i-1] * 0.4 + value[i] * 0.6
			rate[i] = rate[i-1] * 0.3 + rate[i] * 0.7
	start = 0
	prev_mode = mode[0]
	fig,ax = plt.subplots(figsize=(15,4))

	ax.margins(0)
	ax.set_xlabel('episodes')
	ax.set_ylabel('marginal value',color='b')

	ax.plot(epi, value, color='b')
	ax.tick_params(axis='y',labelcolor='b')
	
	# plt.text(0.01, 0.95,'peach : exploration',transform=ax.transAxes)
	# plt.text(0.01, 0.9,'green : exploitation', transform=ax.transAxes)

	for i in range(len(mode)):
		if prev_mode != mode[i]:
			if prev_mode == 0:
				ax.axvspan(epi[start], epi[i], facecolor=[256/256., 236/256., 226/256.], alpha=0.9)
			else:
				ax.axvspan(epi[start], epi[i], facecolor=[224/256., 240/256., 247/256.], alpha=0.9)

			start = i

		prev_mode = mode[i]
	if prev_mode == 0:
		ax.axvspan(epi[start], epi[len(mode)-1], facecolor=[256/256., 236/256., 226/256.], alpha=0.9)
	else:
		ax.axvspan(epi[start], epi[len(mode)-1], facecolor=[224/256., 240/256., 247/256.], alpha=0.9)

	ax2 = ax.twinx()
	ax2.set_ylabel('update rate',color='r')
	ax2.plot(epi, rate, color='r')
	ax2.tick_params(axis='y',labelcolor='r')

	plt.show()

if __name__=="__main__":
	plot()