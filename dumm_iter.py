import os, sys
import numpy as np
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "./mxnet/python"))
import mxnet as mx

# batch class
class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key
        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class dummIter(mx.io.DataIter):
    def __init__(self, batch_size=32, dim=256, length =100):
        self.batch_size = batch_size
        self.dim = dim
        self.length = length
        self.provide_data=[('tc_array', (batch_size, length, dim)), 
                           ('cc_array', (batch_size, length, dim)),
                           ('tw_array', (batch_size, length, dim)),
                           ('cw_array', (batch_size, length, dim))]
        self.provide_label=[('label', (batch_size, 2000))]

    def reset(self):
        pass
    
    def next(self):
	tc_array=np.zeros((self.batch_size, self.length, self.dim))
	cc_array=np.zeros((self.batch_size, self.length, self.dim))
	tw_array=np.zeros((self.batch_size, self.length, self.dim))
	cw_array=np.zeros((self.batch_size, self.length, self.dim))
        label = np.zeros((self.batch_size, 2000))
        data = [mx.nd.array(tc_array), mx.nd.array(tc_array), mx.nd.array(tc_array), mx.nd.array(tc_array)]
        label= [mx.nd.array(label)]
        data_name = ['tc_array', 'cc_array', 'tw_array', 'cw_array']
        label_name = ['label']
        bucket_key =None
        return SimpleBatch(data_name, data, label_name, label, bucket_key)

    def __iter__(self):
	tc_array=np.zeros((self.batch_size, self.length, self.dim))
	cc_array=np.zeros((self.batch_size, self.length, self.dim))
	tw_array=np.zeros((self.batch_size, self.length, self.dim))
	cw_array=np.zeros((self.batch_size, self.length, self.dim))
        label = np.zeros((self.batch_size, 2000))
        data = [mx.nd.array(tc_array), mx.nd.array(tc_array), mx.nd.array(tc_array), mx.nd.array(tc_array)]
        label= [mx.nd.array(label)]
        data_name = ['tc_array', 'cc_array', 'tw_array', 'cw_array']
        label_name = ['label']
        bucket_key =None
        yield SimpleBatch(data_name, data, label_name, label, bucket_key)


if __name__ == '__main__':
    dum = dummIter(32,256,100)
    for d in dum:
        print d.provide_data
        print d.provide_label
        print d.data
        print d.label
        exit()

