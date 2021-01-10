import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from IPython import embed
import random
import time
import math
from sklearn.linear_model import LinearRegression
import re
def plot():

	data = open('vtable')

	epi = []
	dict_list = []
	while True:
		l = data.readline()
		if not l: 
			break
		l = l.split(':')
		e = float(l[1])
		d = dict()

		l = data.readline()
		while l != '\n':
			l = re.split("[(, :)]", l)
			key = float(l[1])
			li = []
			l = l[7:]

			size = math.floor(len(l) / 5)
			for i in range(size):
				li.append([float(l[5*i]), float(l[5*i+1])])
			d[key / 1.5] = li
			l = data.readline()
		epi.append(e)
		dict_list.append(d)


	data.close()

	rows = np.array([12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0])
	
	epi_selected = []
	columns = []

	for i in range(len(epi)-1):	
		epi_selected.append(epi[i])
		vp_dict = dict_list[i]
			# for k, v in zip(vp_dict.keys(), vp_dict.values()):
			# 	print('(', k * 0.5, ',', (k+1)*0.5,') : ',v, ' ', np.array(v).mean(axis=0)[1])
			# print()
		y_predict = []
		for i in range(len(rows)):
			v_key = math.floor(rows[i] * 1 / 1.5) 
			count = 0
			mean = 0
			if v_key in vp_dict:
				for i in range(len(vp_dict[v_key])):
					mean += vp_dict[v_key][i][1]
					count += 1
					
				if count != 0:
					mean /= count	
			y_predict.append(mean)
		columns.append(y_predict)

	fig, ax = plt.subplots(figsize=(20, 5))

	# hide axes
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	bar_width = 0.4

	columns_print = []
	for i in range(len(rows)):
		columns_print_col = []
		for j in range(len(columns)):
			#if columns[j][i] != -1:
				# if i == 0 or columns[j][i-1] == -1:
				# 	bot = 0
				# else:
				# 	bot = columns[j][i-1]
				# diff = columns[j][i] - bot
				# plt.bar(j, diff, bar_width, bottom=bot, color=colors[i])
			columns_print_col.append(round(columns[j][i], 2))
		columns_print.append(columns_print_col)
#	columns = np.array(columns).T
	columns_print = np.array(columns_print)
	columns_print = columns_print[::-1]

	rows = rows[::-1]
	# colors = colors[::-1]
	the_table = plt.table(cellText=columns_print,
                      rowLabels=rows,
                      colLabels=epi_selected,
                      loc='center')
	# the_table.set_fontsize(24)
	plt.show()

if __name__=="__main__":
	plot()