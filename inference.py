import tensorflow as tf 
import pdb

from global_vals import *

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

            pdb.set_trace()
            return cells

    '''
    embedded_x: (bs, BLOCKS, embedding_size)
    '''
    def BiRNN(self, embedded_x, dropout, FLAGS):

        pdb.set_trace()
        embedded_x = tf.unstack(tf.transpose(embedded_x, perm=[1,0,2])) #(BLOCKS, bs, embedding_size)

        fw_cells = self.RNN("fw", dropout, FLAGS)

        bw_cells = self.RNN("bw", dropout, FLAGS)

        outputs, _, _ = tf.nn.static_bidirectional_rnn(fw_cells, bw_cells, embedded_x, dtype=tf.float32)

        return outputs[-1]

    '''
    FLAGS contains parameters
    '''
    def __init__(self, FLAGS):

        self.input_x1 = tf.placeholder(tf.int32, [None, BLOCKS], name="input_x1")

        self.input_x2 = tf.placeholder(tf.int32, [None, BLOCKS], name="input_x2")

        self.input_y  = tf.placeholder(tf.float32, [None], name="input_y")

        self.dropout = tf.placeholder(tf.float32, name="dropout")

        #embedding: seq ==> list of blocks ==> list of int ==> list of vec
        with tf.name_scope("embedding"):

            self.W = tf.Variable(tf.random_uniform([NUMEMBED, FLAGS.embedding_size]))

            self.embedded_x1 = tf.nn.embedding_lookup(self.W, self.input_x1)

            self.embedded_x2 = tf.nn.embedding_lookup(self.W, self.input_x2)

        #BiRNN
        with tf.variable_scope("BiRNN") as scope:

            self.out1 = self.BiRNN(self.embedded_x1, self.dropout, FLAGS)

            scope.reuse_variables()
            pdb.set_trace()

            self.out2 = self.BiRNN(self.embedded_x2, self.dropout, FLAGS)

        #distance
        with tf.name_scope("distance"):
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        #loss
        with tf.name_scope("loss"):
            #self.loss = tf.reduce_sum(tf.square(tf.subtract(self.distance, self.input_y)))/FLAGS.batch_size
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.distance, self.input_y)), name="loss")

        pdb.set_trace()

        return


