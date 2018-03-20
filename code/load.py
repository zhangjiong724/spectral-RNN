# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import numpy as np
import csv
import pickle
import sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
import os
import cPickle as pickle
import urllib2

datasets_dir = os.getcwd() + '/../data/'

def load_mnist_local():
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))


	return np.concatenate((trX,teX)), np.concatenate((trY,teY))




''' prepare dataset '''
def load_mnist(params, permute=False):
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data, mnist.target, random_state=params.random_seed)
    #mnist_X, mnist_y = load_mnist_local()
    mnist_X = mnist_X / 255.0
    print mnist_X.shape, mnist_y.shape
    print("MNIST data prepared")

    mnist_X, mnist_y = mnist_X.astype('float32'), mnist_y.astype('int64')
    if permute:
        np.random.seed(0); permute = np.random.permutation(784)
        mnist_X = mnist_X[:, permute]
    def flatten_img(images):
        '''
        images: shape => (n, rows, columns)
        output: shape => (n, rows*columns)
        '''
        n_rows    = images.shape[1]
        n_columns = images.shape[2]
        for num in range(n_rows):
            if num % 2 != 0:
                images[:, num, :] = images[:, num, :][:, ::-1]
        output = images.reshape(-1, n_rows*n_columns)
        return output

    time_steps = 28*28
    if len(params.dataset) > 6: # mnist.xx
        time_steps = int(params.dataset.split('.')[1])
    mnist_X = mnist_X.reshape((-1, time_steps, 28*28/time_steps))
    #mnist_X = flatten_img(mnist_X) # X.shape => (n_samples, seq_len)
    print "mnist_X.shape = ", mnist_X.shape
    #mnist_X = mnist_X[:, :, np.newaxis] # X.shape => (n_samples, seq_len, n_features)
    mnist_y_one_hot = np.zeros((mnist_y.shape[0], 10))
    for i in xrange(len(mnist_y)):
        mnist_y_one_hot[i][mnist_y[i]] = 1
    print "mnist_y.shape = ", mnist_y_one_hot.shape

    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y_one_hot,
                                                        test_size=0.2,
                                                        random_state=params.random_seed)
    # need to set parameters according to dataset
    params.time_steps = train_X.shape[1]
    params.input_size = train_X.shape[2]
    params.output_size = 10
    params.regression_flag = False
    return train_X, test_X, train_y, test_y


def adding_task(params, fname=datasets_dir+'Adding_task/data', ntrain=50000, ntest=1000):
    filename = fname + str(params.time_steps)
    data = np.loadtxt(filename, delimiter=',').astype(np.float32)
    x = data[:,1:]; y = data[:,0]
    assert(ntrain+ntest <= x.shape[0])
    train_X = x.reshape((x.shape[0], x.shape[1]//2, 2))
    train_Y = y.reshape((y.shape[0], 1))
    params.time_steps = train_X.shape[1]
    params.input_size = train_X.shape[2]
    params.output_size = 1
    params.regression_flag = True
    print("Adding task with %i time step prepared!"%params.time_steps)
    print "Adding X shape: ", train_X.shape
    print "Adding Y shape: ", train_Y.shape

    return train_X[0 : ntrain], train_X[ntrain : ntrain + ntest], train_Y[0 : ntrain], train_Y[ntrain : ntrain + ntest]

