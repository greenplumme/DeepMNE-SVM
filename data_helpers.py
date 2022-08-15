# -*- coding: utf-8 -*-
# !/usr/bin/env python
# author:xuehansheng(xhs1892@gmail.com)

from pickletools import float8
import numpy as np
from sklearn import preprocessing
import pickle
yeast_nums = 16668 #11094 #13460 #7563 #6818 16933 #

def load_genes(org):
	if org == 'Pakinson':
		genes = readGenes('../data/networks/Pakinson_string_genes.txt')
	return genes

def load_networks(org):
	num_nets = 6
	str_nets = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']
	adj_nets = np.zeros((num_nets, yeast_nums, yeast_nums))
	for idx in range(num_nets):
		path_net = '../data/networks/'+org +'/'+org+'_string_'+str_nets[idx]+'_adjacency.txt'
		adj_nets[idx] = readNetworks(path_net)
	return adj_nets

def load_musks():
	num_musks = 2
	musks = ['../data/rwr/Net13/Net1_U_musk.txt','../data/rwr/Net13/Net3_U_musk.txt']
	adj_musks = np.zeros((num_musks, yeast_nums, yeast_nums),dtype='float16')
	for idx in range(num_musks):
		# path_fusion = '/media/userdisk1/jjpeng/xuehansheng/'+org+'_'+str_fusions[idx]+'_rwr.txt'
		path_musk = musks[idx]
		# adj_fusions[idx] = readRWR(path_fusion)
		# normalize
		# print(readRWR(path_mask).shape)
		adj_musks[idx] = np.transpose(readRWR(path_musk))
	return adj_musks

def load_fusions():
	num_fusions = 2
	fusions = ['../data/rwr/Net13/Net1_U_0.4_rwr.txt','../data/rwr/Net13/Net3_U_0.4_rwr.txt']
	adj_fusions = np.zeros((num_fusions, yeast_nums, yeast_nums),dtype='float16')
	for idx in range(num_fusions):
		# path_fusion = '/media/userdisk1/jjpeng/xuehansheng/'+org+'_'+str_fusions[idx]+'_rwr.txt'
		path_fusion = fusions[idx]
		# adj_fusions[idx] = readRWR(path_fusion)
		# normalize
		print(readRWR(path_fusion).shape)
		adj_fusions[idx] = np.transpose(preprocessing.scale(readRWR(path_fusion)))
	return adj_fusions

def readGenes(filepath):
	dataSet = []
	for line in open(filepath):
		line = line.strip()
		dataSet.append(line)
	return dataSet

def readNetworks(filepath):
	print(filepath)
	network = np.zeros([yeast_nums, yeast_nums],dtype='float16')
	for line in open(filepath):
		line = line.strip()
		temp = list(map(str,line.split('	')))
		network[int(temp[0])-1, int(temp[1])-1] = temp[2]
	return network

def readRWR(filepath):
	network = np.zeros([yeast_nums,yeast_nums],dtype='float16')
	import joblib
	data = joblib.load(open(filepath,'rb'))#,encoding='bytes')
	#network = [[float(v) for v in line.rstrip('\n').split('\t')] for line in data]
	network = [[float(v) for v in line] for line in data]
	return np.array(network)

def write_encoded_file(data, file_path):
	with open(file_path, "w") as f:
		for line in data:
			tempLine = ""
			for i in range(len(line)):
				tempLine = tempLine + str(line[i]) + "\t"
			tempLine = tempLine + "\n"
			f.write('%s'% tempLine)

def read_encoded_file(file_path):
    network = []
    for line in open(file_path):
        line = line.strip()
        temp = list(map(float,line.split("\t")))
        network.append(temp)
    return np.array(network)