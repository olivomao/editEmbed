import pdb
import numpy as np

from global_vals import *

def batch_eval_iter(x1, x2, y):

    x1 = np.asarray(x1)
    
    x2 = np.asarray(x2)
    
    y  = np.asarray(y)

    for bn in range(len(x1)):

        stt = bn
        stp = bn+1

        yield x1[stt: stp], x2[stt: stp], y[stt: stp]

def batch_iter(x1, x2, y, FLAGS):

    x1 = np.asarray(x1)
    
    x2 = np.asarray(x2)
    
    y  = np.asarray(y)

    num_batches_per_epoch = int(len(x1)/FLAGS.batch_size)+1

    for ep in range(FLAGS.num_epochs):

        for bn in range(num_batches_per_epoch):

            stt = bn * FLAGS.batch_size
            stp = min((bn+1)*FLAGS.batch_size, len(x1))

            yield x1[stt: stp], x2[stt: stp], y[stt: stp]

'''
input: st
output: [b_0, ..., b_BLOCKS-1] each is substring of st (zero padded and chopped into BLOCKLEN)
'''
def transform(st):

    assert len(st) <= MAXLEN

    #pdb.set_trace()

    st = st + "0"*(MAXLEN-len(st))

    transformed_st = [int(st[i*BLOCKLEN: (i+1)*BLOCKLEN],2) for i in range(BLOCKS)]    

    #pdb.set_trace()

    #binary str to int

    return transformed_st

'''
read file and output x1,x2 and y
- x1[i] and x2[i] are list of blocks representing binary seqs,
- y[i] is their relevant edit distance
'''
def load(file):

    x1 = []
    x2 = []
    y = []

    x1_str = []
    x2_str = []

    with open(file, 'r') as f:

        for line in f:

            if line[0]=='#': continue

            tokens = line.strip().split('\t')

            a = tokens[0]#x1[i]
            x1.append(transform(a))
            x1_str.append(a)

            b = tokens[1]#x2[i]
            x2.append(transform(b))
            x2_str.append(b)

            c = tokens[2]#y[i]
            y.append(int(c))
            #pdb.set_trace()

    return x1, x2, y, x1_str, x2_str
