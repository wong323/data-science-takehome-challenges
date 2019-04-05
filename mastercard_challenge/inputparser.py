
import pandas as pd
import hashlib as hashy
#import mxnet as mx
#from mxnet.gluon import nn
#from mxnet import nd


# note that this configuration does not include the class label (see the helper function targets_maker below)
# it also doesn't include the accountid_hashed field, due to the difficulty of using this hash as a feature
# I'd like to improve this but need a second opinion as to how to include this feature!
config_fields = [
'is_fraud',
'clicks',
'pw_speed_burst_a',
'pw_speed_burst_b', 
'pw_speed_burst_c',
'pw_time', 
'pw_pos_a', 
'pw_pos_b', 
'pw_pos_speed_a',
'pw_pos_speed_b',
'pw_pos_speed_d',
'pw_time_a',
'pw_time_b',
'un_speed_burst_a',
'un_speed_burst_b',
'un_time',
'un_speed',
'un_pos_b',
'un_speed_burst_a_st',
'un_speed_burst_c',
'un_speed_burst_b_st',
'un_speed_burst_h',
'un_time_b']
    

def csv_reader(input):
    a = pd.read_csv(input)
    return a

def ndarray_maker(input):
    a = nd.array(input[[*config_fields]].values)
    return a

def x_maker(input):
    x = input[[*config_fields]]
#    return a
    return x

def ndarray_targets_maker(input):
    a = nd.array(input[['is_fraud']].values)
    return a
