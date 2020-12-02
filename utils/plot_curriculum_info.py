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
		epi.append(float(l[1]))
		mode.append(int(l[2]))
		l = data.readline()
		l = l.split(':')
		value.append(float(l[0]))
		rate.append(float(l[1]))

	data.close()
	return epi, mode, value, rate

def plot():
	epi, mode, value, rate = read_vratio_type1('curriculum_info10')
	start = 0
	prev_mode = mode[0]

	fig,ax = plt.subplots(figsize=(15,5))

	ax.margins(0)
	ax.set_xlabel('episodes')
	ax.set_ylabel('marginal value',color='blue')

	ax.plot(epi, value, color='blue')
	ax.tick_params(axis='y',labelcolor='blue')
	
	plt.text(0.01, 0.95,'peach : exploration',transform=ax.transAxes)
	plt.text(0.01, 0.9,'green : exploitation', transform=ax.transAxes)

	for i in range(len(mode)):
		if prev_mode != mode[i]:
			if prev_mode == 0:
				ax.axvspan(epi[start], epi[i], facecolor='xkcd:pale peach', alpha=0.6)
			else:
				ax.axvspan(epi[start], epi[i], facecolor='xkcd:light olive', alpha=0.6)

			start = i

		prev_mode = mode[i]
	if prev_mode == 0:
		ax.axvspan(epi[start], epi[len(mode)-1], facecolor='xkcd:pale peach', alpha=0.6)
	else:
		ax.axvspan(epi[start], epi[len(mode)-1], facecolor='xkcd:light olive', alpha=0.6)

	ax2 = ax.twinx()
	ax2.set_ylabel('update rate',color='red')
	ax2.plot(epi, rate, color='red')
	ax2.tick_params(axis='y',labelcolor='red')

	plt.show()

if __name__=="__main__":
	plot()