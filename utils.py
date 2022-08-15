# !/usr/bin/env python
# coding=utf-8
# author:xuehansheng(xhs1892@gmail.com)

import numpy as np
import seaborn as sns
# import scipy.spatial.distance as ssd
import operator
from functools import reduce
from scipy.stats import pearsonr
from data_helpers import load_musks,write_encoded_file
yeast_nums =16668 #16933 #  13460 # 11094

def extractConstraints(representation,idx_net):
    num_ml = 0
    num_cl = 0
    d = []
    musks = load_musks()
    mustlink_matrix = np.zeros((yeast_nums, yeast_nums))
    cannotlink_matrix = np.zeros((yeast_nums, yeast_nums))
    # # 0.05
    # upper = [0.7078,0.501]
    # lower = [0.0249,0.068]
    # 0.1
    upper = [0.697,0.489]
    lower = [0.0506,0.1495]
    M = np.zeros((yeast_nums, yeast_nums))
    for i in range(len(representation)):
        for j in range(i):
            distance = pearsonr(representation[i],representation[j])[0]
            d.append(distance)
            M[i,j] = distance
            if distance <= lower[idx_net]:
                cannotlink_matrix[i,j] = 1.0
                cannotlink_matrix[j,i] = 1.0
                num_cl = num_cl + 1
            if distance >= upper[idx_net]:
                mustlink_matrix[i,j] = 1.0
                mustlink_matrix[j,i] = 1.0
                num_ml = num_ml + 1
    d = np.array(d)
    sns.set_style('darkgrid')
    sns_plot = sns.displot(d).figure
    sns_plot.savefig("output"+str(idx_net)+".png")

    ml_mat = np.array(musks[idx_net]) * np.array(mustlink_matrix)
    cl_mat = np.array(musks[idx_net]) * np.array(cannotlink_matrix)

    path = 'Net13/net'+str(idx_net)+'_pcc.txt'
    write_encoded_file(np.array(M), path)
    # write_encoded_file(np.array(cannotlink_matrix), path[1])

    print('extract constraints:', num_ml, num_cl)
    print("###### lower{},upper:{},lower0.1{},upper0.1{}".format(lower,upper,np.percentile(np.absolute(M), 10),np.percentile(np.absolute(M), 90)))
    return ml_mat, cl_mat

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

class SemiDataset:
    def __init__(self, data, constraints, constraints_):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._constraints = constraints
        self._constraints_ = constraints_
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    @property
    def constraints(self):
        return self._constraints

    @property
    def constraints_(self):
        return self._constraints_


    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self._data = self.data[idx]
            self._constraints = self.constraints[idx]
            self._constraints_ = self.constraints_[idx]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            start1 = start
            end1 = self._num_examples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end =  self._index_in_epoch

            constraints_rest_part_1 = self.constraints[start1:end1, start1:end1]
            constraints_rest_part_2 = self.constraints[start1:end1, start:end]
            constraints_rest_part = np.hstack((constraints_rest_part_1, constraints_rest_part_2))

            constraints_rest_part_1_ = self.constraints_[start1:end1, start1:end1]
            constraints_rest_part_2_ = self.constraints_[start1:end1, start:end]
            constraints_rest_part_ = np.hstack((constraints_rest_part_1_, constraints_rest_part_2_))

            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            self._data = self.data[idx0]
            self._constraints = self.constraints[idx0]
            self._constraints_ = self.constraints_[idx0]

            data_new_part =  self._data[start:end]  
            constraints_new_part_1 = self._constraints[start:end, start:end]
            constraints_new_part_2 = self._constraints[start:end, start1:end1]
            constraints_new_part = np.hstack((constraints_new_part_1, constraints_new_part_2))

            constraints_new_part_1_ = self._constraints_[start:end, start:end]
            constraints_new_part_2_ = self._constraints_[start:end, start1:end1]
            constraints_new_part_ = np.hstack((constraints_new_part_1_, constraints_new_part_2_))

            data_batch = np.concatenate((data_rest_part, data_new_part), axis=0)
            constraints_batch = np.concatenate((constraints_rest_part, constraints_new_part), axis=0)
            constraints_batch_ = np.concatenate((constraints_rest_part_, constraints_new_part_), axis=0)
            return data_batch, constraints_batch, constraints_batch_
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._constraints[start:end, start:end], self._constraints_[start:end, start:end]
