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
    def __init__(self, args, iterator=None, iterator_infer=None):
        self.args = args

        self.iterator = iterator #BatchedInput
        self.iterator_infer = iterator_infer

        with tf.variable_scope("graph", dtype=tf.float32) as scope: 
            
            self.embedding_encoder, \
            self.embedding_decoder, \
            self.vocab_size = self.create_emb(args.seq_type,
                                              args.blocklen,
                                              args.embed_size,
                                              "embeddings",
                                              tf.float32)

            self.enc_cell = self.build_cell(args.enc_num_layers,
                                            args.enc_num_units,
                                            args.enc_forget_bias,
                                            "enc_cell")

            self.dec_cell = self.build_cell(args.dec_num_layers,
                                            args.dec_num_units,
                                            args.dec_forget_bias,
                                            "dec_cell")
            
            self.output_layer = layers_core.Dense(self.vocab_size, \
                                                  use_bias=False, \
                                                  name="decoder/output_projection")
            
            self.encoder_emb_input, \
            self.encoder_outputs, \
            self.encoder_state = self.build_encoder(self.iterator,
                                                    self.embedding_encoder,
                                                    self.enc_cell,
                                                    "train",
                                                    dtype=tf.float32)
           
            self.encoder_emb_input_infer, \
            self.encoder_outputs_infer, \
            self.encoder_state_infer = self.build_encoder(self.iterator_infer,
                                                          self.embedding_encoder,#shared
                                                          self.enc_cell,#shared
                                                          "infer",
                                                          dtype=tf.float32)

            self.outputs, \
            self.sample_id, \
            self.final_context_state, \
            self.logits = self.build_decoder(self.args,
                                             self.iterator,
                                             self.embedding_decoder,
                                             self.encoder_outputs,
                                             self.encoder_state,
                                             self.dec_cell,
                                             "train",
                                             outer_layer=self.output_layer,#used at infer
                                             dtype=tf.float32)

            self.outputs_infer, \
            self.sample_id_infer, \
            self.final_context_state_infer, \
            self.logits_infer = self.build_decoder(self.args,
                                                   self.iterator_infer,
                                                   self.embedding_decoder,
                                                   self.encoder_outputs,
                                                   self.encoder_state,
                                                   self.dec_cell,
                                                   "infer",
                                                   outer_layer=self.output_layer,#used at infer
                                                   dtype=tf.float32)


            self.loss = self.compute_loss(self.iterator, \
                                          self.logits, \
                                          "train")

            #logits and target_output have different dimensions, can't calculate now
            #calculated in calc_validation_loss using Python
            #
            #self.loss_infer = self.compute_loss(self.iterator_infer, \
            #                                    self.logits_infer, \
            #                                    "infer")
        
        return

    '''
    create embedding for encoder and decoder
    '''
    def create_emb(self, seq_type,
                         blocklen,
                         embed_size,
                         scope="embeddings",
                         dtype=tf.float32):

        if seq_type == 0:
            vocab_size = pow(2, blocklen)+3 #including <unk> <s> and </s>
        else:
            vocab_size = pow(2, blocklen*2)+3 #dna, blocklen doubled
            
        embed_size = embed_size 
        with tf.variable_scope(scope, dtype=dtype) as scope:

            emb_enc = tf.get_variable("shared_emb", [vocab_size, embed_size], dtype=dtype)
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
    def compute_loss(self, iterator, logits, purpose):

        print("compute_loss for %s"%purpose)

        def get_max_time(tensor): #(bs, max_time, etc)
            return tensor.shape[1].value or tf.shape(tensor)[1]

        #pdb.set_trace()
        target_output = iterator.target_output

        max_time = get_max_time(target_output)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        
        mask = tf.sequence_mask(iterator.target_sequence_length, max_time, dtype=logits.dtype)
        
        crossent = crossent * mask # some predictions at time t are not valid
        
        current_batch_size = tf.shape(iterator.source)[0] 
        
        loss = tf.reduce_sum(crossent / tf.to_float(current_batch_size))
        
        return loss


    def build_encoder(self, 
                      iterator,
                      embedding_encoder,
                      enc_cell, 
                      purpose,
                      dtype=tf.float32):

        print('build encoder for %s'%purpose)

        with tf.variable_scope("%s/encoder"%purpose, dtype=dtype) as scope:

            source = iterator.source #[batch_size, max_time], per element should be an id number transferred from a block of binary or dna sequence
            
            encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source) #[batch_size, max_time, num_units]

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(enc_cell,
                                                               encoder_emb_inp,
                                                               dtype=dtype,
                                                               sequence_length=iterator.source_sequence_length) 
        return encoder_emb_inp, encoder_outputs, encoder_state

    '''
    build an rnn cell (single or multi layers) for encoder or decoder
    '''
    def build_cell(self, num_layers, num_units, forget_bias, name):

        cell_list = self.create_cell_list(num_layers,
                                          num_units,
                                          forget_bias)

        if len(cell_list)==1:
            res = cell_list[0]
        else:
            res = tf.contrib.rnn.MultiRNNCell(cell_list)

        #res = tf.identity(res, name=name)
        return res

    def create_cell_list(self, num_layers, num_units, forget_bias):

        cell_list = []
        for i in range(num_layers):
            single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
            cell_list.append(single_cell)

        return cell_list

    '''
    output:
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
    def build_decoder(self, 
                      args,
                      iterator,
                      embedding_decoder,
                      encoder_outputs,
                      encoder_state,
                      dec_cell,
                      purpose,
                      outer_layer,#used at infer, None for train
                      dtype=tf.float32):

        print('build decoder for %s'%purpose)

        with tf.variable_scope("%s/decoder"%purpose, dtype=dtype) as decoder_scope:
            
            decoder_initial_state = encoder_state

            if purpose=="train":

                target_input = iterator.target_input

                decoder_emb_input = tf.nn.embedding_lookup(embedding_decoder, target_input)

                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_input,
                                                           iterator.target_sequence_length)

                my_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, decoder_initial_state) 

                #dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope=decoder_scope)

                sample_id = None #outputs.sample_id may exceed vocab range

                logits = outer_layer(outputs.rnn_output)

                #sample_id = tf.Print(sample_id,
                #                     [tf.shape(iterator.source),
                #                      tf.shape(logits),
                #                      tf.shape(iterator.target_output),
                #                     ],
                #                  "train shapes of src, logits, tgt_output")



            elif purpose=="infer":
            
                #pdb.set_trace()
                current_batch_size = args.batch_size #args.n_clusters_validation #tf.shape(iterator.source)[0] 
                start_tokens = tf.fill([current_batch_size], 1) #starting with <s> 
                end_token = 2 #ending with </s>
                
                max_encoder_length = tf.reduce_max(iterator.source_sequence_length)
                max_decoder_length = tf.to_int32(tf.round(tf.to_float(max_encoder_length)*4.0))

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder,
                                                                  start_tokens,
                                                                  end_token)
                decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,#reuse decoder cell
                                                          helper,
                                                          decoder_initial_state,
                                                          output_layer=outer_layer #per timestamp used
                                                          )
                outputs, final_context_state, _  = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                     maximum_iterations=max_decoder_length,
                                                                                     scope=decoder_scope)
                logits = outputs.rnn_output
                sample_id = outputs.sample_id

                #sample_id = tf.Print(sample_id,
                #                  [tf.shape(iterator.source),
                #                   tf.shape(iterator.target_output),
                #                   tf.shape(logits)],
                #                  "shapes of src, tgt_output, logits")

        return outputs, sample_id, final_context_state, logits 



