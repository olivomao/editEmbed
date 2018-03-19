import tensorflow as tf 
from argparse import ArgumentParser
import pdb
import sys
import numpy as np
from Bio import SeqIO

#from global_vals import *
#from proc_data import *
from edit_dist import edit_dist

class siamese:

    '''
    return a one side RNN (several layers)
    '''
    def RNN(self, scope, dropout, FLAGS):

        with tf.name_scope(scope), tf.variable_scope(scope):
            stacked_rnn = []
            for _ in range(FLAGS.num_layers):
                cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_sz, forget_bias=1.0, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
                stacked_rnn.append(cell)
            cells = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)

            #pdb.set_trace()
            return cells

    '''
    embedded_x: (bs, BLOCKS, embedding_size)
    '''
    def BiRNN(self, embedded_x, dropout, FLAGS):

        #pdb.set_trace()
        embedded_x = tf.unstack(tf.transpose(embedded_x, perm=[1,0,2])) #(BLOCKS, bs, embedding_size)

        fw_cells = self.RNN("fw", dropout, FLAGS)

        bw_cells = self.RNN("bw", dropout, FLAGS)

        outputs, _, _ = tf.nn.static_bidirectional_rnn(fw_cells, bw_cells, embedded_x, dtype=tf.float32)

        return outputs[-1]

    '''
    FLAGS contains parameters
    '''
    def __init__(self, FLAGS):

        #pdb.set_trace()
        #store into the model for future usage
        self.maxlen = tf.Variable(FLAGS.maxlen, name="maxlen")
        self.blocklen = tf.Variable(FLAGS.blocklen, name="blocklen")
        blocks = FLAGS.maxlen / FLAGS.blocklen
        numembed = pow(2,FLAGS.blocklen)

        self.input_x1 = tf.placeholder(tf.int32, [None, blocks], name="input_x1")

        self.input_x2 = tf.placeholder(tf.int32, [None, blocks], name="input_x2")

        self.input_y  = tf.placeholder(tf.float32, [None], name="input_y")

        self.dropout = tf.placeholder(tf.float32, name="dropout")        

        #embedding: seq ==> list of blocks ==> list of int ==> list of vec
        with tf.name_scope("embedding"):

            self.W = tf.Variable(tf.random_uniform([numembed, FLAGS.embedding_size]))

            self.embedded_x1 = tf.nn.embedding_lookup(self.W, self.input_x1)

            self.embedded_x2 = tf.nn.embedding_lookup(self.W, self.input_x2)

        #BiRNN
        with tf.variable_scope("BiRNN") as scope:

            self.out1 = self.BiRNN(self.embedded_x1, self.dropout, FLAGS)

            scope.reuse_variables()
            #pdb.set_trace()

            self.out2 = self.BiRNN(self.embedded_x2, self.dropout, FLAGS)

        #distance
        with tf.name_scope("distance"):
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        #loss
        with tf.name_scope("loss"):
            #self.loss = tf.reduce_sum(tf.square(tf.subtract(self.distance, self.input_y)))/FLAGS.batch_size
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.distance, self.input_y)), name="loss")

        #pdb.set_trace()

        return

'''
===== FLAGS:
- FLAGS.seq_type #0: binary 1: DNA/ATCG

- FLAGS.model_prefix #e.g. model_dir/ckpt

- FLAGS.allow_soft_placement
- FLAGS.log_device_placement

===== members:
- self.seq_type #0: binary 1: ATCG
- self.model_prefix #e.g. model_dir/ckpt
- self.allow_soft_placement
- self.log_device_placement

- self.sess
- self.graph

- self.input_x1
- self.input_x2
- self.dropout

- self.distance

===== note:
class used to make predictions based on trained model:
- load a trained model
- predict distance of seq pair based on trained model
'''
class Predict:

    def restore_session(self):
        
        ckpt = self.model_prefix #e.g. model_dir/ckpt

        session_conf = tf.ConfigProto(allow_soft_placement=self.allow_soft_placement, log_device_placement=self.log_device_placement)

        sess = tf.Session(config=session_conf)

        saver = tf.train.import_meta_graph("%s.meta"%ckpt) #tf.train.Saver()

        sess.run(tf.global_variables_initializer())    

        saver.restore(sess, ckpt)

        return sess

    '''
    seq_type: 1 - DNA/ATCG seq
    '''
    def predict(self, seq_1, seq_2):

        if self.seq_type==1:
            seq_1 = "".join([dna2bin[seq_1[i]] for i in range(len(seq_1))])
            seq_2 = "".join([dna2bin[seq_2[i]] for i in range(len(seq_2))])

        seq_1 = np.asarray(transform(seq_1)); seq_1 = np.reshape(seq_1, (1,len(seq_1)));
        seq_2 = np.asarray(transform(seq_2)); seq_2 = np.reshape(seq_2, (1,len(seq_2)));

        predicted_dist = self.sess.run([self.distance], feed_dict={ self.input_x1: seq_1,
                                                               self.input_x2: seq_2,
                                                               self.dropout:  1.0
                                                             })
        #pdb.set_trace() 
        return predicted_dist[0]

    '''
    transformed_seq_1 and transformed_seq_2 are not strs but transformed by seq2nn for model's direct usage
    '''
    def predict0(self, transformed_seq_1, transformed_seq_2):
        predicted_dist =  self.sess.run(self.distance, feed_dict={ self.input_x1: transformed_seq_1,
                                                                     self.input_x2: transformed_seq_2,
                                                                     self.dropout:  1.0
                                                                   })
        return predicted_dist[0]

    def __init__(self, seq_type,
                       model_prefix,
                       allow_soft_placement=True,
                       log_device_placement=False):

        self.seq_type = seq_type
        self.model_prefix = model_prefix
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement

        self.graph = tf.Graph()

        with self.graph.as_default():

            self.sess = self.restore_session()

            self.input_x1 = self.graph.get_operation_by_name("input_x1").outputs[0]
            self.input_x2 = self.graph.get_operation_by_name("input_x2").outputs[0]
            self.dropout = self.graph.get_operation_by_name("dropout").outputs[0]
            # Tensors we want to evaluate
            self.distance = self.graph.get_operation_by_name("distance/distance").outputs[0]

            #pdb.set_trace()


            self.maxlen = self.graph.get_operation_by_name("maxlen").outputs[0].eval(session=self.sess)#500 #later to be read from trained model
            self.blocklen = self.graph.get_operation_by_name("blocklen").outputs[0].eval(session=self.sess)#10 #later to be read from trained model
            #print(self.maxlen)
            #print(self.blocklen)
            #pdb.set_trace()

        return   

'''
Test function
'''

def test_Predict(args):

    pdb.set_trace()

    pdt = Predict(args.seq_type,
                  args.model_prefix,
                  args.allow_soft_placement,
                  args.log_device_placement)
   
    s2n_obj = seq2nn(pdt.seq_type, pdt.maxlen, pdt.blocklen)
 
    #simulate_data_binary
    #note: when do transform, block len & max len should be consistent with trained model
    seq_1="CGACGGACTTTTAGGTTCAGCTGGCTGCTTCCAATACATCGTCTCTCCCGACCTGTATACTTGCTTACTTCCAAAATGCGCACCTCCTGGCCCGCTTTCACGACTGAGATGCTTAGACAAGTGCTCTTAGATTCCGACGCGAGTGTAGTT"
    seq_2="CGACGGACTTTTAGGTTCAGCTGGCTCTTCCAATACATCGTTCTCCCGACCTGTATACTTGCTTACTTCCAAAATGCGCACCTCCTGCGCCCGCTTTCACGACTGAGATGCTTAGCACATGTGCTCTTAGATATCCGACGCGAGTGTAGTT"

    e_dst = edit_dist(seq_1, seq_2) 

    #predicted_dist = pdt.predict(seq_1, seq_2)
    #print("seq_1=%s\nseq_2=%s\nedit_dist=%f\npredicted_distance=%f\n"%(seq_1, seq_2, e_dst, predicted_dist))

    x1 = s2n_obj.transform(seq_1)
    x2 = s2n_obj.transform(seq_2)
    predicted_dist0 = pdt.predict0(x1,x2)
    pdb.set_trace()
    return

'''
This is a class to transfer seq into nn input
- to be used by both training and prediction
'''
class seq2nn:

    def __init__(self, seq_type, maxlen, blocklen):
        self.seq_type = seq_type        
        self.maxlen = maxlen
        self.blocklen = blocklen
        self.blocks = maxlen/blocklen
        self.numembed = pow(2,blocklen)        

    '''
    input: st
    output: [b_0, ..., b_BLOCKS-1] each is substring of st (zero padded and chopped into BLOCKLEN)
    '''
    def transform_binary(self, st):

        assert len(st) <= self.maxlen

        st = st + "0"*(self.maxlen-len(st))

        transformed_st = [int(st[i*self.blocklen: (i+1)*self.blocklen],2) for i in range(self.blocks)]    

        return transformed_st

    def dna2bin_str(self, dna_str):
        dna2bin_dic = {'A':"00", 'T':"01", 'C':"10", 'G':"11"}
        bin_str = "".join([dna2bin_dic[dna_str[i]] for i in range(len(dna_str))])
        #print('%s to %s'%(dna_str, bin_str))
        return bin_str

    #in shape of (1,len(seq))
    def transform(self, seq_str):

        #print('transform: seq_type='+str(self.seq_type)+' python_type='+str(type(self.seq_type)))

        if self.seq_type==1:
            #print('to do dna2bin_str')
            seq_str1 = self.dna2bin_str(seq_str)  
        else:
            #print('skip dna2bin_str')
            seq_str1 = seq_str     

        #print('seq_str1: '+seq_str1)

        seq = np.asarray(self.transform_binary(seq_str1))
        #pdb.set_trace()
        seq = np.reshape(seq, (1, len(seq)))

        return seq

    '''
    this is used in calc dist using learned model,
    to transform all seqs into nn-compatible ones,
    to avoid transform seq strs during pairwise seq calculation

    input: general fa_file
    output: list of objests (description,  seq/transformed one)
    '''
    def transform_seqs_from_fa(self, fa_file):
        seqs = list(SeqIO.parse(fa_file, 'fasta'));
        #fa_tseq_list = [Fa_tSeq(seq.description, seq.seq, self.transform(str(seq.seq))) for seq in seqs]
        #pdb.set_trace()
        fa_tseq_list = []
        for seq in seqs:
            fa_tseq_list.append(Fa_tSeq(seq.description, seq.seq, self.transform(str(seq.seq))))
            #print(seq.description)
            #print(str(seq.seq))
            
        return fa_tseq_list

'''
transformed Seq obj from Fa
'''
class Fa_tSeq:
    def __init__(self, dscrpt, seq_str, tSeq):
        self.description = dscrpt
        self.seq = seq_str
        self.tseq = tSeq
        
    

     

if __name__ == "__main__":

    parser = ArgumentParser()
    subs = parser.add_subparsers()

    '''
    test functions of class Predict
    '''
    if sys.argv[1]=="test_Predict":
        s_parser = subs.add_parser("test_Predict")
        s_parser.add_argument('--seq_type', default=0, type=int, help="0: binary 1:dna/ATCG")
        s_parser.add_argument('--model_prefix', type=str, help="prefix of trained model")
        s_parser.add_argument('--allow_soft_placement', default=True, type=bool)
        s_parser.add_argument('--log_device_placement', default=False, type=bool)
        args = parser.parse_args()
        test_Predict(args)
    else:
        pdb.set_trace()
