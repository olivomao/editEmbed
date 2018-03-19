import pdb
import numpy as np
from Bio import SeqIO

#from global_vals import *
from util import *
from inference import seq2nn

def batch_eval_iter(x1, x2, y):

    x1 = np.asarray(x1)
    
    x2 = np.asarray(x2)
    
    y  = np.asarray(y)

    for bn in range(len(x1)):

        stt = bn
        stp = bn+1

        yield x1[stt: stp], x2[stt: stp], y[stt: stp]

'''
FLAGS <==> args (we use args)
'''
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

'''
read file and output x1,x2 and y
- x1[i] and x2[i] are list of blocks representing binary seqs,
- y[i] is their relevant edit distance

seq_type==1: ATCG to be transformed into binary
'''
def load(args): #(file, seq_type=0):

    file = args.train_input 
    seq_type = args.seq_type

    #pdb.set_trace()
    s2n_obj = seq2nn(args.seq_type, args.maxlen, args.blocklen)

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
            x1_str.append(a)

            #pdb.set_trace()
            a = s2n_obj.transform(a).flatten()
            x1.append(a)
            #
            #if seq_type==1:
            #    a = "".join([dna2bin[a[i]] for i in range(len(a))])
            #x1.append(transform(a))
            #

            b = tokens[1]#x2[i]
            x2_str.append(b)

            b = s2n_obj.transform(b).flatten()
            x2.append(b)
            #
            #if seq_type==1:
            #    b = "".join([dna2bin[a[i]] for i in range(len(b))])
            #x2.append(transform(b))
            #

            c = tokens[2]#y[i]
            y.append(int(c))
            #pdb.set_trace()

    return x1, x2, y, x1_str, x2_str

'''
prepare training data (in sampled_seq_fa and pairwise_distance)  generated using simulate_data.py

seq_type: 0 (binary seq) and 1 (dna/ATCG seq)

max_num_to_sample: -1 no limit
'''
dna2bin = {"A":"00", "T":"01", "C":"10", "G":"11"}

def load2(args): #sampled_seq_fa, pairwise_dist, seq_type, max_num_to_sample=-1):

    sampled_seq_fa = args.train_input1
    pairwise_dist = args.train_input2
    seq_type = args.seq_type
    max_num_to_sample = args.max_num_to_sample

    maxlen = args.maxlen
    blocklen = args.blocklen 

    s2n_obj = seq2nn(seq_type, maxlen, blocklen)

    #collect id and seq pairs
    #pdb.set_trace()
    dic_id_seq = {} #key: seq_id, val: seq {binary or binary transformed from ATCG}

    seqs = s2n_obj.transform_seqs_from_fa(sampled_seq_fa) #list(SeqIO.parse(sampled_seq_fa, 'fasta'))
    for seq in seqs:
        seq_id = seq.description.split()[0]
        dic_id_seq[seq_id] = seq.tseq.flatten()

    #prepare seq pairs
    N_tot = sum([1 for line in open(pairwise_dist, 'r')])
    #pdb.set_trace()
    iterCnt = iterCounter(N_tot, "check pos/neg samples")
    n_tp = [0, 0, 0] #pair type 0,1,2
    with open(pairwise_dist, 'r') as fin:
        for line in fin:
            iterCnt.inc()
            line = line.strip()
            if line[0]=='#': continue
            tp = int(line.split()[2])
            n_tp[tp] = n_tp[tp]+1
    iterCnt.finish()
    logPrint("(pair_type, cnts)=(0/same, %d),(1/diff, %d),(2, %d)"%(n_tp[0], n_tp[1], n_tp[2]))

    #sampling
    #pdb.set_trace()
    x1 = []
    x2 = []
    y  = []
    iterCnt = iterCounter(N_tot, "sample training data" )
    with open(pairwise_dist, 'r') as fin:
        for line in fin:
            iterCnt.inc()
            line = line.strip()
            if line[0]=='#': continue
            if max_num_to_sample!=-1 and len(x1)>max_num_to_sample:
                break 
            tokens = line.split()
            s1 = dic_id_seq[tokens[0]] #ATCG transformed to bin
            x1.append(s1)
            s2 = dic_id_seq[tokens[1]]
            x2.append(s2)
            y.append(float(tokens[3]))
    iterCnt.finish()
    logPrint("training data sampled")
    #pdb.set_trace()
    return x1, x2, y, [], [] #last two items correspond to x1_str and x2_str but deprecated

'''
#backup
def load2(sampled_seq_fa, pairwise_dist, seq_type, max_num_to_sample=-1):

    #collect id and seq pairs
    pdb.set_trace()
    dic_id_seq = {} #key: seq_id, val: seq {binary or binary transformed from ATCG}

    seqs = list(SeqIO.parse(sampled_seq_fa, 'fasta'))
    for seq in seqs:
        seq_id = seq.description.split()[0]
        seq_seq = str(seq.seq)
        if seq_type==1:
            seq_seq = "".join([dna2bin[seq_seq[i]] for i in range(len(seq_seq))])
        dic_id_seq[seq_id] = seq_seq

    #prepare seq pairs
    N_tot = sum([1 for line in open(pairwise_dist, 'r')])
    pdb.set_trace()
    iterCnt = iterCounter(N_tot, "check pos/neg samples")
    n_tp = [0, 0, 0] #pair type 0,1,2
    with open(pairwise_dist, 'r') as fin:
        for line in fin:
            iterCnt.inc()
            line = line.strip()
            if line[0]=='#': continue
            tp = int(line.split()[2])
            n_tp[tp] = n_tp[tp]+1
    iterCnt.finish()
    logPrint("(pair_type, cnts)=(0/same, %d),(1/diff, %d),(2, %d)"%(n_tp[0], n_tp[1], n_tp[2]))

    #sampling
    pdb.set_trace()
    x1 = []
    x2 = []
    y  = []
    x1_str = []
    x2_str = []
    iterCnt = iterCounter(N_tot, "sample training data" )
    with open(pairwise_dist, 'r') as fin:
        for line in fin:
            iterCnt.inc()
            line = line.strip()
            if line[0]=='#': continue
            if max_num_to_sample!=-1 and len(x1)>max_num_to_sample:
                break 
            tokens = line.split()
            s1 = dic_id_seq[tokens[0]] #ATCG transformed to bin
            x1.append(transform(s1))
            x1_str.append(s1)
            s2 = dic_id_seq[tokens[1]]
            x2.append(transform(s2))
            x2_str.append(s2)
            y.append(float(tokens[3]))
    iterCnt.finish()
    logPrint("training data sampled")
    pdb.set_trace()
    return x1, x2, y, x1_str, x2_str
'''

