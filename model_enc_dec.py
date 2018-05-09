'''
an enc-dec architecture, modified based on tensorflow's nmt tutorial
'''

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import pdb

class Model(object):

    '''
    args: Namespace type e.g. args.num_encoder_layers=3
    '''
    def __init__(self, args, iterator):
        self.args = args
        self.iterator = iterator #BatchedInput
        self.embedding_encoder, self.embedding_decoder, self.vocab_size = self.create_emb()
        self.output_layer = layers_core.Dense(self.vocab_size, use_bias=False, name="decoder/output_projection")
        return

    '''
    create embedding for encoder and decoder
    '''
    def create_emb(self):

        if self.args.seq_type == 0:
            vocab_size = pow(2, self.args.blocklen)+3 #including <unk> <s> and </s>
        else:
            vocab_size = pow(2, self.args.blocklen*2)+3 #dna, blocklen doubled
            
        embed_size = self.args.embed_size 
        with tf.variable_scope("embeddings", dtype=tf.float32) as scope:

            emb_enc = tf.get_variable("shared_emb", [vocab_size, embed_size], dtype=tf.float32)
            emb_dec = emb_enc

        '''
        pdb.set_trace()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(emb_enc))
        pdb.set_trace()
        '''

        return emb_enc, emb_dec, vocab_size

    '''
    define outputs of encoder, decoder and loss

    For training, src, tgt_input, tgt_output are given
    For validation, src, tgt_output are given, decoder don't use tgt_output
    Depends on training or validation, decoder and loss are different
    '''
    def build_graph(self, scope=None):

        print("build_graph")

        dtype = tf.float32

        with tf.variable_scope(scope or "graph", dtype=dtype):

            self.encoder_emb_input, self.encoder_outputs, self.encoder_state = self.build_encoder()
            self.decoder_emb_input, self.decoder_outputs, self.sample_id,  self.decoder_state, self.logits = self.build_decoder(self.encoder_outputs, self.encoder_state) 
            self.loss = self.compute_loss(self.logits)
        return
    
    def compute_loss(self, logits):

        print("compute_loss")

        def get_max_time(tensor): #(bs, max_time, etc)
            return tensor.shape[1].value or tf.shape(tensor)[1]

        #pdb.set_trace()
        target_output = self.iterator.target_output
        max_time = get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=self.logits)
        mask = tf.sequence_mask(self.iterator.target_sequence_length, max_time, dtype=self.logits.dtype)
        crossent = crossent * mask # some predictions at time t are not valid
        loss = tf.reduce_sum(crossent / tf.to_float(self.args.batch_size))
        return loss


    def build_encoder(self):

        print('build encoder')

        with tf.variable_scope("encoder", dtype=tf.float32) as scope:

            source = self.iterator.source #[batch_size, max_time], per element should be an id number transferred from a block of binary or dna sequence
            
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, source) #[batch_size, max_time, num_units]

            cell = self.build_cell(self.args.enc_num_layers,
                                   self.args.enc_num_units,
                                   self.args.enc_forget_bias)
            
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell,
                                                               encoder_emb_inp,
                                                               dtype=tf.float32,
                                                               sequence_length=self.iterator.source_sequence_length) 
        return encoder_emb_inp, encoder_outputs, encoder_state

    '''
    build an rnn cell (single or multi layers) for encoder or decoder
    '''
    def build_cell(self, num_layers, num_units, forget_bias):

        cell_list = self.create_cell_list(num_layers,
                                          num_units,
                                          forget_bias)

        if len(cell_list)==1:
            return cell_list[0]
        else:
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def create_cell_list(self, num_layers, num_units, forget_bias):

        cell_list = []
        for i in range(num_layers):
            single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
            cell_list.append(single_cell)

        return cell_list

    '''
    output:
    - decoder_emb_input (bs, max_time, embed_size)
    - decoder_outputs
      rnn_output (bs, max_time, last dec_hdsz)
      sample_id  (bs, max_time) sampled from rnn_output
    - sample_id
    - decoder_state
      For LSTM State Tuple = (c,h), where
                                          c (bs, last dec_hdsz)
                                          h (bs, last dec_hdsz)
    - logits (bs, max_time, vocab_size) not softmax  
    '''
    def build_decoder(self, encoder_outputs, encoder_state):

        print('build decoder')

        with tf.variable_scope("decoder", dtype=tf.float32) as decoder_scope:
            cell = self.build_cell(self.args.dec_num_layers,
                                   self.args.dec_num_units,
                                   self.args.dec_forget_bias
                                  )
            decoder_initial_state = self.encoder_state

            #========== for training
            target_input = self.iterator.target_input

            decoder_emb_input = tf.nn.embedding_lookup(self.embedding_decoder, target_input)

            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_input,
                                                       self.iterator.target_sequence_length)

            my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state) 

            #dynamic decoding
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope=decoder_scope)

            sample_id = outputs.sample_id

            logits = self.output_layer(outputs.rnn_output)

            #========== For inference
            pdb.set_trace()
            start_tokens = tf.fill([self.args.n_cluster_validation], 1) #starting with <s> 
            end_token = 2 #ending with </s>
            #infer_maximum_iterations = self.args.cluster_len * 4
            #max_encoder_length = tf.reduce_max()
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                                    start_tokens,
                                                                    end_token)
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell,#reuse decoder cell
                                                            infer_helper,
                                                            decoder_initial_state,
                                                            output_layer=self.output_layer #per timestamp used
                                                           )
            infer_outputs, infer_final_context_state, _  = tf.contrib.seq2seq.dynamic_decode(infer_decoder,
                                                                                            maximum_iterations=infer_maximum_iterations,)

        #pdb.set_trace()
        return decoder_emb_input, outputs, sample_id, final_context_state, logits 



