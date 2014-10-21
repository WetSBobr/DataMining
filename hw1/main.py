# ! /usr/bin/env python
import argparse

import numpy as np
import pylab
import numpy.random
import time
import sklearn as sk
from sklearn.metrics import mutual_info_score
from sklearn.metrics import silhouette_score

import warnings

warnings.simplefilter("error")
warnings.filterwarnings("ignore", 'Mean of empty slice.')
warnings.filterwarnings("ignore", 'invalid value encountered in double_scalars')

__author__ = 'Bovbel Ilya'


def main():
	args = parse_args ()

	my_data = np.genfromtxt(args.file, delimiter=',', skip_header=1, usecols=(2,3,4,5,6,7), dtype=int)
	data_true = np.genfromtxt(args.file, delimiter=',', skip_header=1, usecols=(1), dtype=int)
	
	lables = DBSCAN (my_data, args.minPts, args.eps)
	
	print "number of clasters = ", np.unique (lables).size
	print "purity = ", purity (lables, data_true)
	print "rand index = ", rand_index (lables, data_true)
	print "mutual info = ", mutual_info_score (lables, data_true)
	print silhouette_score (my_data, lables)
	pass



def parse_args():
	parser = argparse.ArgumentParser(description='DBSCAN clustering')
	parser.add_argument('-q', dest='quality', choices=['purity', 'rand_index', 'mutual_information'], default='purity', help='Criteria for quality clustering')
	parser.add_argument('-f', dest='file', default='data.csv', nargs=1)
	
	parser.add_argument('-e', dest='eps',  type=float, default=40000, help='maximum radius of sfera, where you can find neighbors')
	parser.add_argument('-m', dest='minPts',  type=int, default=6, help='minimum number of elements, when you take an element for a center')
	return parser.parse_args()


def DBSCAN (X, minPts, eps):
	size = X.shape[0]
	lables = np.zeros (size)
	
	for i in xrange (size):
		if lables[i] != 0:
			continue
		neighbors = regionQuery (i, X, eps)
		if neighbors.size < minPts:
			lables[i] = -1
		else:
			C = int (np.max (lables) + 1)
			lables[i] = C
			lables = expandCluster (X[i], X, neighbors, C, eps, minPts, lables)
	
	return lables
	
def expandCluster(P, X, NeighborPts, C, eps, minPts, lables):
	for i in NeighborPts:
		if lables[i] == -1 or lables[i] == 0:
			lables[i] =  C
		if lables[i] != 0:
			continue

		neighbors = regionQuery (i, X, eps)
		if neighbors.shape[0] >= minPts:
			NeighborPts = np.hstack ([NeighborPts, neighbors])	
	return lables

def regionQuery(j, X, eps):
	P = X[j]
	neighbors = np.array ([j])
	
	for i in xrange (X.shape[0]):
		length = np.sqrt (np.sum((P-X[i])*(P-X[i])))
		if (length < eps):
			neighbors = np.append (neighbors, i)
			
	return neighbors
		
def rand_index (lables, data_true):
	
	a = 0
	b = 0
	
	size = int (np.min ([lables.size, data_true.size]))
	for i in xrange (size):
		for j in xrange (size):
			if lables [i] == lables [j] and data_true [i] == data_true [j]:
				a = a+1
			if lables [i] != lables [j] and data_true [i] != data_true [j]:
				b=b+1
	return 1.0 * (a+b) / (size*(size-1) )

def purity (lables, data_true):
	u_labels = np.unique(lables)
	u_data_true = np.unique(data_true)
	
	size = int (np.min ([lables.size, data_true.size]))
	max_elements = 0
	lable_elements = np.arange (size)
	for i in u_labels:
		cur_max = 0;
		xl = lable_elements[lables == i]
		for j in u_data_true:
			xd = lable_elements[data_true == j]
			x = np.intersect1d (xd, xl).size
			if x > cur_max:
				cur_max = x
		max_elements += cur_max
			
	return 1.0*max_elements/size

if __name__ == '__main__':
	main()




