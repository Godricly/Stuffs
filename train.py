import os, sys 
import numpy as np
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "./mxnet/python"))
import mxnet as mx
from cudnn_iter import dummIter
import sym

dum = dummIter(32)
net = sym.sym_gen(100,100,100,100)

mod = mx.mod.Module(net,
                      data_names=['tc_array', 'cc_array', 'tw_array', 'cw_array'],
                      label_names=['label'], context=mx.gpu(0))
mod.bind(data_shapes = dum.provide_data, label_shapes = dum.provide_label)#,
#         data_names=['tc_array', 'cc_array', 'tw_array', 'cw_array'],
#         label_names=['label'])
mod.init_params()
mod.fit(dum, num_epoch=100)
