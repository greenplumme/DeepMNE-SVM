import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime
import numpy as np
import tensorflow as tf
from models import *
from sklearn import model_selection
from sklearn.model_selection import *
from sklearn.model_selection import KFold
from data_helpers import *

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS #name,value,comment
flags.DEFINE_string('org', 'Pakinson', 'Dataset string')  # 'yeast', 'human'
flags.DEFINE_integer('net_dims', 16668, 'Dimensional number of input networks')
flags.DEFINE_integer('net_nums', 2, 'Number of input networks')
flags.DEFINE_list('learning_rate', [[1e-3,0.01], [1e-4,1e-4]], 'Initial learning rate')
# flags.DEFINE_float('learning_rate', [0.1, 0.01, 0.01], 'Initial learning rate')
flags.DEFINE_integer('batch_size', 64, 'Initial batch size')
# flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_list('iter_num', [[2000,5000], [2000,2000]], 'Number of iterations to train')
# flags.DEFINE_integer('iter_num', [10000, 1000, 1000], 'Number of iterations to train')
flags.DEFINE_integer('layers_num', 2, 'Number of the whole model')
flags.DEFINE_list('hidden_dim', [1600, 500, 500], 'Number of units in hidden layers')
# flags.DEFINE_integer('hidden_dim', [3200, 1600, 500], 'Number of units in hidden layers.')
# flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
# flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
# flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('optimizer', 'Adam', 'Optimizer')
flags.DEFINE_float('gamma', 0.5, 'The weight of mustlink constraints')
flags.DEFINE_float('alpha', 0.5, 'The weight of cannotlink constraints')
flags.DEFINE_float('percent', 0.05, 'Pecentage of extracting constraints')

# load data
print ('Loading networks... ')
# fusions = load_fusions()

input_dim = FLAGS.net_dims
mustlinks = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.net_dims))
cannotlinks = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.net_dims))
constraints_ml = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.net_dims))
constraints_cl = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.net_dims))
str_nets = ['./Net13/net1_'+str(FLAGS.iter_num[0][0])+'_'+str(FLAGS.hidden_dim[0])+'_'+str(FLAGS.learning_rate[0][0])+'.txt', 
			'./Net13/net3_'+str(FLAGS.iter_num[0][1])+'_'+str(FLAGS.hidden_dim[0])+'_'+str(FLAGS.learning_rate[0][1])+'.txt']


for idx_layer in range(FLAGS.layers_num):
	emb = np.zeros((FLAGS.net_nums, FLAGS.net_dims, FLAGS.hidden_dim[idx_layer]))

	if idx_layer == 0:
		for idxx in range(FLAGS.net_nums):
			print(str_nets[idxx])
			emb[idxx] = read_encoded_file(str_nets[idxx])
	else:
		for idx_net in range(FLAGS.net_nums):
			semiAE = SemiAutoEncoder(input_dim, FLAGS.hidden_dim[idx_layer], FLAGS.learning_rate[idx_layer][idx_net], FLAGS.batch_size, FLAGS.iter_num[idx_layer][idx_net], FLAGS.gamma, FLAGS.alpha)
			emb[idx_net] = semiAE.train(fusions[idx_net], constraints_ml[idx_net], constraints_cl[idx_net], FLAGS.optimizer)
	
	# if idx_layer == 0:
	# 	for idxx in range(FLAGS.net_nums):
	# 		temp_path = './'+str_nets[idxx]
	# 		write_encoded_file(emb[idxx], temp_path)

	if idx_layer != FLAGS.layers_num - 1:
		print ('Extracting constraints...')
		for idx_net in range(FLAGS.net_nums):
			temp_mustlink, temp_cannotlink = extractConstraints(emb[idx_net],idx_net)
			mustlinks[idx_net] = temp_mustlink
			cannotlinks[idx_net] = temp_cannotlink

		print ('Merging constraints...')
		for idx in range(FLAGS.net_nums):
			temp_mustlink = np.zeros((FLAGS.net_dims, FLAGS.net_dims))
			temp_cannotlink = np.zeros((FLAGS.net_dims, FLAGS.net_dims))
			for idxx in range(FLAGS.net_nums):
				if idxx != idx:
					temp_mustlink = temp_mustlink + mustlinks[idxx]
					temp_cannotlink = temp_cannotlink + cannotlinks[idxx]
			constraints_ml[idx] = temp_mustlink 
			constraints_cl[idx] = temp_cannotlink
			print(len(constraints_ml[idx].nonzero()[0]) / 2, len(constraints_cl[idx].nonzero()[0]) / 2)

	input_dim = FLAGS.hidden_dim[idx_layer]
	fusions = emb

# output embedding
str_nets = ['./Net13/net1_'+str(FLAGS.iter_num[0][0])+'_'+str(FLAGS.iter_num[1][0])+'_'+str(FLAGS.hidden_dim[0])+'_'+str(FLAGS.hidden_dim[1])+'_'+str(FLAGS.learning_rate[0][0])+'_'+str(FLAGS.learning_rate[0][1])+'_.txt', 
			'./Net13/net3_'+str(FLAGS.iter_num[0][1])+'_'+str(FLAGS.iter_num[1][1])+'_'+str(FLAGS.hidden_dim[0])+'_'+str(FLAGS.hidden_dim[1])+'_'+str(FLAGS.learning_rate[0][1])+'_'+str(FLAGS.learning_rate[1][1])+'_.txt']

for idxx in range(FLAGS.net_nums):
	temp_path = './'+str_nets[idxx]
	write_encoded_file(emb[idxx], temp_path)
