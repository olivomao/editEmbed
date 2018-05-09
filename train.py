import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb
import os
#pdb.set_trace()
import tensorflow as tf 
import datetime
import time
from argparse import ArgumentParser, Namespace
import numpy as np
import collections #BatchedInput

#from global_vals import *
from proc_data import *
import inference
from tensorflow.python import debug as tf_debug
from util import logPrint, convert_str_to_bool_int_float
from model_enc_dec import Model


'''

There're two source types:
(1) input_type==0: based on simulate_binary
    tain_input is in binary_seq_pair format.
(2) input_type==1: based on simulate_data
    train_input1: sampled_seq_fa format
    train_input2: pairwise_distance format

'''
def prepare_data(args):
    #pdb.set_trace()

    #load training data
    if args.input_type==0:
        #pdb.set_trace()
        x1, x2, y, _, _ = load(args)
    else: #input_type==1
        #pdb.set_trace()
        x1, x2, y, _, _ = load2(args)

    x1_tr, x2_tr, y_tr, x1_vld, x2_vld, y_vld = split_train_validation(x1, x2, y)

    #pdb.set_trace()

    batches = batch_iter(x1_tr, x2_tr, y_tr, args)

    n_samples = len(x1_tr)

    return batches,n_samples, (x1_vld, x2_vld, y_vld)

t2d = {0: 'binary', 1: 'DNA'}
'''
for dna seq, blocklen will be doubled, because per A,T,C,G will be described by 2 bits each.

- header disabled because tf.table will read them in
'''
def prepare_vocab(vocab_path, seq_type, blocklen):

    with open(vocab_path, 'w') as vf:

        #header = ''
        #header += '## type:%s\n'%t2d[seq_type]
        #header += '## blocklen:%d\n'%blocklen

        #vf.write(header)

        vf.write('<unk>\n')
        vf.write('<s>\n')
        vf.write('</s>\n')

        if seq_type==1:
            blocklen = blocklen * 2

        for i in range(pow(2,blocklen)):
            vf.write(("{0:0%db}"%blocklen).format(i)+'\n') 
        logPrint('%s written'%vocab_path)

    return

'''
currently used in seq2seq architecture. to be processed further by tf.dataset and iterator

e.g.
seq=0011, bl=2 return '00 11'
seq=ATCG, bl=3 return '00,01,10,11' ==> '000 110 11[0]'
'''
dna2bin = {'A':'00', 'T':'01', 'C':'10', 'G':'11'}

def get_seq2blocks(seq, seq_type, blocklen):

    if seq_type==1:
        blocklen = 2*blocklen
        seq = ''.join(dna2bin[seq[i]] for i in range(len(seq)))

    seq_len = len(seq)
    if seq_len % blocklen != 0:
        seq = seq.ljust(seq_len + blocklen - seq_len % blocklen, '0')
        seq_len = len(seq)

    n_blocks = seq_len / blocklen

    return ' '.join([seq[i*blocklen:(i+1)*blocklen] for i in range(n_blocks)])


'''
input:  args
output: args.model_dir_path/data_processed/s_i1.seq2ids
                                          /s_i2.seq2ids
                                          /vocat.txt
        to be used by tf.dataset and iterator
'''
#def prepare_data_seq2seq(args):

'''
input: seq2seq_path (can be train or validation)
       seq_type, blocklen
output: s_i1_pah, s_i2_path

in particular, (x,x') seq pair in seq2seq_path split, with blocks of x into s_i1_path
                                                      with blocks of x' into s_i2_path
'''
def prepare_data_seq2seq(seq2seq_path,
                         seq_type,
                         blocklen,
                         s_i1_path,
                         s_i2_path):

    #pdb.set_trace() 

    #run_cmd('mkdir -p %s'%os.path.join(args.model_dir_path, 'data_processed'))

    #s_i1_path =  os.path.join(args.model_dir_path, 'data_processed', 's_i1.seq2ids')
    #s_i2_path =  os.path.join(args.model_dir_path, 'data_processed', 's_i2.seq2ids')
    #vocab_path = os.path.join(args.model_dir_path, 'data_processed', 'vocab.txt')

    #prepare_vocab(vocab_path, args.seq_type, args.blocklen)
    #pdb.set_trace()

    with open(seq2seq_path, 'r') as in_f, \
         open(s_i1_path, 'w') as out_f1, \
         open(s_i2_path, 'w') as out_f2:

        header =  '## src:%s\n'%seq2seq_path
        header +=  '## type:%s\n'%t2d[seq_type]
        header +=  '## blocklen:%d\n'%blocklen

        out_f1.write(header)
        out_f2.write(header)
       
        for line in in_f:

            if line[0]=='#': continue

            tokens = [itm for itm in line.strip().split() if itm!='']

            if len(tokens)<2: continue

            s_i1 = tokens[0]
            s_i2 = tokens[1]

            s_i1_blocks_str = get_seq2blocks(s_i1, seq_type, blocklen)
            s_i2_blocks_str = get_seq2blocks(s_i2, seq_type, blocklen)

            out_f1.write(s_i1_blocks_str+'\n')
            out_f2.write(s_i2_blocks_str+'\n')

        logPrint('%s written'%s_i1_path)
        logPrint('%s written'%s_i2_path)

    #pdb.set_trace()
    return s_i1_path, s_i2_path

'''
input is log file (i, train_loss, validation_loss)

output is log fig (x-axis is i, y-axis contains 2 curves: train_loss and validation loss)
'''
def plot_log(input_log, output_fig):

    #pdb.set_trace()

    logPrint('[plot_log] starts')

    #load data
    list_i = []
    list_tr_loss = []
    list_vld_loss = []

    with open(input_log, 'r') as fi:

        for line in fi:

            if line!='' and line[0]=='#': continue

            tokens = line.strip().split()
            if tokens==[]: continue

            list_i.append(int(tokens[0]))
            list_tr_loss.append(float(tokens[1]))
            list_vld_loss.append(float(tokens[2]))

    #pdb.set_trace()

    fig, ax = plt.subplots()
    ax.plot(list_i, list_tr_loss, label='training loss') #marker='o', 
    ax.plot(list_i, list_vld_loss, label='validation loss') #marker='o', 
    ax.legend()

    #show/save all
    plt.tight_layout()
    plt.savefig(output_fig) #plt.show()
    #pdb.set_trace()

    logPrint('[plot_log] finished. %s written'%output_fig)

    return

'''
input:
FLAGS      contains TF parameters for training; same as args (we use args)
batches    to yield batch data for training
n_samples: # of training samples
vld_data:  (x1_vld, x2_vld, y_vld), extracted from raw training data (e.g. 15%) used for validation
debug:     only used (True) for debug purpose

output:
model stored at FLAGS.ckpt_prefix
'''
def train(FLAGS,
          batches,
          n_samples,
          vld_data,
          debug=False
          ):

    #load model
    siamese = inference.siamese(FLAGS)

    #optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    grads_and_vars = optimizer.compute_gradients(siamese.loss)

    capped_gvs = []
    for grad, var in grads_and_vars:
        if grad is not None:
            #capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
            capped_gvs.append((tf.clip_by_norm(grad, 1.), var))
        else:
            capped_gvs.append((grad, var))

    tr_op_set = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    #session
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    sess = tf.InteractiveSession(config=session_conf) #tf.Session(config=session_conf)

    #
    saver = tf.train.Saver()

    #actual train
    sess.run(tf.global_variables_initializer())

    if FLAGS.load_model==True:
        saver.restore(sess, FLAGS.ckpt_prefix)
    
    #pdb.set_trace()
    n_batches = n_samples/FLAGS.batch_size * FLAGS.num_epochs

    #prepare validation data
    x1_vld, x2_vld, y_vld = vld_data
    x1_vld = np.asarray(x1_vld)
    x2_vld = np.asarray(x2_vld)
    y_vld  = np.asarray(y_vld)

    logPath = FLAGS.train_output_dir+'/log.txt'
    logFigPath = FLAGS.train_output_dir+'/log.png' 

    #'''
    logFile = open(logPath, 'w'); logFile.write('#i\ttrain_loss\tvalidation_loss\n')

    if debug==True: sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    for i in range(n_batches):

        try:
            b_x1, b_x2, b_y = batches.next()
        except:
            pdb.set_trace()

        _, b_step, b_loss = sess.run([tr_op_set, global_step, siamese.loss], feed_dict={siamese.input_x1: b_x1,
                                                                                        siamese.input_x2: b_x2,
                                                                                        siamese.input_y:  b_y, 
                                                                                        siamese.dropout:  FLAGS.dropout})

        
        #pdb.set_trace()
        vld_loss = sess.run([siamese.loss], feed_dict={  siamese.input_x1: x1_vld,
                                                         siamese.input_x2: x2_vld,
                                                         siamese.input_y:  y_vld, 
                                                         siamese.dropout:  1})
        vld_loss = vld_loss[0]

        if np.isnan(b_loss) or np.isnan(vld_loss):
            print('train loss %s or valid loss %s is nan at i=%d'%(b_loss, vld_loss, i))
            if logFile.closed==False: logFile.close()
            plot_log(logPath, logFigPath)
            #pdb.set_trace()
            return #pdb.set_trace()

        if True: #b_step % 20 == 0:
            print('i=%d, train_loss=%f, validation_loss=%f'%(i, b_loss, vld_loss))
            logFile.write('%d\t%f\t%f\n'%(i, b_loss, vld_loss))

        if b_step % 100 == 0:
            try:
                print("save model b_step=%d"%b_step)
                saver.save(sess, FLAGS.train_output_dir+'ckpt_step_%d_loss_%s'%(b_step, str(b_loss)))
            except:
                print("saver.save exception at b_step=%d"%(b_step))
                if logFile.closed==False: logFile.close()
                plot_log(logPath, logFigPath)
                return

    #pdb.set_trace()
    if logFile.closed==False: logFile.close()
    #'''
    pdb.set_trace()

    plot_log(logPath, logFigPath) 
    return

class BatchedInput(
        collections.namedtuple("BatchedInput",
                               ("initializer", "source", "target_input", "target_output", "source_sequence_length", "target_sequence_length"))):
    pass

'''
Based on s_i1_path (x, in block chunks), s_i2_path (transformed_x, in block chunks) and vocab_path (including <unk> <s> and </s>), 

We build a lookup table from vocab and then transform x and transformed_x from block chunks into ids.
In addition, [1]+tgt ==> tgt_input and tgt+[2] ==> tgt_output, and all cols padded (w/ 2, eos) to have same length across batch.

output an iterator (src_ids, tgt_input, tgt_output, src_len/before padding, tgt_len/before padding) 

'''
def prepare_minibatch_seq2seq(s_i1_path, s_i2_path, vocab_path, args):
    
    pdb.set_trace()
    #used for words/blocks to ids
    table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocab_path,
                                                    default_value=0, #<unk> defined in vocab_path
                                                   )
    
    src_dataset = tf.data.TextLineDataset(s_i1_path)
    src_dataset = src_dataset.filter(lambda line: tf.logical_and(tf.not_equal(tf.substr(line, 0, 1), "#"),
                                                                 tf.not_equal(line, '')))
    tgt_dataset = tf.data.TextLineDataset(s_i2_path)
    tgt_dataset = tgt_dataset.filter(lambda line: tf.logical_and(tf.not_equal(tf.substr(line, 0, 1), "#"),
                                                                 tf.not_equal(line, '')))
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(args.output_buffer_size, args.random_seed)
    src_tgt_dataset = src_tgt_dataset.repeat()

    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values)
                                         ).prefetch(args.output_buffer_size)

    #words/blocks to ids
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (tf.cast(table.lookup(src), tf.int32),tf.cast(table.lookup(tgt), tf.int32))).prefetch(args.output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src, tf.concat(([1], tgt), 0), tf.concat((tgt, [2]), 0))).prefetch(args.output_buffer_size) #1: start of sentence and 2: end of sentence
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in))).prefetch(args.output_buffer_size)

    src_tgt_dataset = src_tgt_dataset.padded_batch(args.batch_size,
                                                   padded_shapes=(tf.TensorShape([None]),
                                                                  tf.TensorShape([None]),
                                                                  tf.TensorShape([None]),
                                                                  tf.TensorShape([]),
                                                                  tf.TensorShape([])),
                                                   padding_values=(2,2,2,0,0)) #padding tgt to same lengths; additional dim added to all cols of src_tgt_dataset
    #pdb.set_trace()

    iterator = src_tgt_dataset.make_initializable_iterator() #src_tgt_dataset.batch(args.batch_size).make_initializable_iterator()
    
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = iterator.get_next()

    #remove outer dimension
    #src_ids = tf.squeeze(src_ids, axis=[0]); print(src_ids)
    #tgt_input_ids = tf.squeeze(tgt_input_ids, axis=[0]); #print(tgt_input_ids)
    #tgt_output_ids = tf.squeeze(tgt_output_ids, axis=[0]); #print(tgt_output_ids)
    #src_seq_len = tf.squeeze(src_seq_len, axis=[0]); #print(src_seq_len)
    #tgt_seq_len = tf.squeeze(tgt_seq_len, axis=[0]); #print(tgt_seq_len)
    #pdb.set_trace()

    bi = BatchedInput(initializer=iterator.initializer,
                      source=src_ids,
                      target_input=tgt_input_ids,
                      target_output=tgt_output_ids,
                      source_sequence_length=src_seq_len,
                      target_sequence_length=tgt_seq_len)
    
    '''
    with tf.Session() as sess:

        #sess.run(iterator.initializer)
        sess.run(bi.initializer)
        sess.run(tf.tables_initializer())

        i=0
        for j in range(args.n_epoch):

            while True:
                try:
                    #x = sess.run(iterator.get_next())
                    x,tgt_i,tgt_o,sl,tl = sess.run([bi.source, bi.target_input, bi.target_output, bi.source_sequence_length, bi.target_sequence_length ])
                    print('ep=%d b=%d %s\n%s'%(j, i, type(x), str(x)))
                    i+=1

                    pdb.set_trace()
                except tf.errors.OutOfRangeError:
                    print('except')
                    sess.run(bi.initializer)
                    break

    pdb.set_trace()
    '''

    return bi #BatchedInput

'''
train seq2seq architecture for one set of hyparam

param_dic is a dic of {param:val} obtained from config file
param_dic transferred to args (Namespace) for easier usage and compatibility

input:
param_dic['seq2seq_pair_path'] = seq2seq_pair
and param_dic['seq2seq_pair_path_validation']

output:
model_dir/ckpt...
where param_dic['model_dir_path'] = model_dir (including model/param)

deprecated from train():
batches    to yield batch data for training
n_samples: # of training samples
vld_data:  (x1_vld, x2_vld, y_vld), extracted from raw training data (e.g. 15%) used for validation
debug:     only used (True) for debug purpose

output:
model stored at FLAGS.ckpt_prefix
'''
def train_seq2seq(param_dic):

    for k,v in param_dic.items():
        param_dic[k] = convert_str_to_bool_int_float(v)
    args = Namespace(**param_dic)

    #process data for tf data pipeline
    run_cmd('mkdir -p %s'%os.path.join(args.model_dir_path, 'data_processed'))

    vocab_path = os.path.join(args.model_dir_path, 'data_processed', 'vocab.txt')
    prepare_vocab(vocab_path, args.seq_type, args.blocklen)
    
    s_i1_path =  os.path.join(args.model_dir_path, 'data_processed', 's_i1.seq2ids')
    s_i2_path =  os.path.join(args.model_dir_path, 'data_processed', 's_i2.seq2ids')
    s_i1_path, s_i2_path = prepare_data_seq2seq(args.seq2seq_pair_path,
                                                args.seq_type,
                                                args.blocklen,
                                                s_i1_path,
                                                s_i2_path)

    s_i1_path_vld =  os.path.join(args.model_dir_path, 'data_processed', 's_i1_validation.seq2ids')
    s_i2_path_vld =  os.path.join(args.model_dir_path, 'data_processed', 's_i2_validation.seq2ids')
    s_i1_path_vld, s_i2_path_vld = prepare_data_seq2seq(args.seq2seq_pair_path_validation,
                                                args.seq_type,
                                                args.blocklen,
                                                s_i1_path_vld,
                                                s_i2_path_vld)


    print('TO IMPROVE TRAIN'); pdb.set_trace()
    batched_input   = prepare_minibatch_seq2seq(s_i1_path,
                                               s_i2_path,
                                               vocab_path,
                                               args)
    model = Model(args, batched_input)
    model.build_graph()
    #pdb.set_trace()
    
    '''
    #check variables
    print('DEBUG - CHECK VARIABLES'); pdb.set_trace()
    vs= tf.trainable_variables()
    print('There are %d trainable variables'%len(vs))
    for v in vs:
        print(v)
    
    print('DEBUG - CHECK NODES'); pdb.set_trace()
    with tf.Session() as sess:

        sess.run(tf.tables_initializer())
        sess.run(batched_input.initializer)
        sess.run(tf.global_variables_initializer())

        while True:
            try:
                #check nodes
                #x = sess.run(batched_input.source)
                #print(x)
                #x = sess.run(model.embedding_encoder)
                #print(x)
                #check decoder outputs
                #pdb.set_trace()
                decoder_emb_input, decoder_outputs, sample_id, decoder_state, logits, loss = sess.run([model.decoder_emb_input, model.decoder_outputs, model.sample_id, model.decoder_state, model.logits, model.loss])
                print(loss)
                #pdb.set_trace()
            except:
                break
    print('DEBUG ENDS'); pdb.set_trace()
    '''
    pdb.set_trace()
    
    #optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    grads_and_vars = optimizer.compute_gradients(model.loss)

    capped_gvs = []
    for grad, var in grads_and_vars:
        if grad is not None:
            #capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
            capped_gvs.append((tf.clip_by_norm(grad, args.grad_max), var))
        else:
            capped_gvs.append((grad, var))

    tr_op_set = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    pdb.set_trace()

    #session
    with tf.Session() as sess:

        #logging
        saver = tf.train.Saver()
        logPath = args.model_dir_path+'/log.txt'
        logFigPath = args.model_dir_path+'/log.png' 
        logFile = open(logPath, 'w'); logFile.write('#i\ttrain_loss\tvalidation_loss\n')

        #actual train
        sess.run(tf.tables_initializer())
        sess.run(batched_input.initializer)
        sess.run(tf.global_variables_initializer())

        #if debug==True: sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        
        for i in range(args.n_batches):

            #decoder_emb_input, decoder_outputs, sample_id, decoder_state, logits, loss = sess.run([model.decoder_emb_input, model.decoder_outputs, model.sample_id, model.decoder_state, model.logits, model.loss])
            _, batch_step, batch_loss = sess.run([tr_op_set, global_step, model.loss])
 
            if np.isnan(batch_loss):
                print('train loss %s is nan at i=%d'%(batch_loss, i))
                if logFile.closed==False: logFile.close()
                #plot_log(logPath, logFigPath)
                return

            print('i=%d (batch_step=%d) , train_loss=%f'%(i,batch_step, batch_loss))

            if i % 100 == 0:
                try:
                    print("save model i=%d"%i)
                    saver.save(sess, args.model_dir_path+'ckpt_step_%d_loss_%s'%(i, str(batch_loss)))
                except:
                    print("saver.save exception at i=%d"%(i))
                    if logFile.closed==False: logFile.close()
                    #plot_log(logPath, logFigPath)
                    return

        #pdb.set_trace()
        if logFile.closed==False: logFile.close()
        #pdb.set_trace()
        #plot_log(logPath, logFigPath) 
    print('TRAIN FINISH'); pdb.set_trace()
    return



if __name__ == "__main__":
    
    parser = ArgumentParser()
    subs = parser.add_subparsers()

    if sys.argv[1]=='use_simulate_binary': #use data generated by simulate_binary
        s_parser = subs.add_parser('use_simulate_binary')
        #IO
        s_parser.add_argument('--input_type', default=0, type=int) #use simulate_binary data
        s_parser.add_argument('--train_input', type=str)
        s_parser.add_argument('--seq_type', type=int, help="0: binary 1: ATCG")
        s_parser.add_argument('--train_output_dir', type=str, help="dir path to store trained model")
        #train
        s_parser.add_argument('--batch_size', default=100, type=int)
        s_parser.add_argument('--num_epochs', default=500, type=int)
        s_parser.add_argument('--load_model', default=0, type=int, help="whether to load somewhat trained model (1) or not (0, default)")
        #model
        s_parser.add_argument('--maxlen', default=500, type=int, help="max len for input sequence")
        s_parser.add_argument('--blocklen', default=10, type=int, help="block len to segment input sequence")

        s_parser.add_argument('--embedding_size', default=200, type=int)
        s_parser.add_argument('--num_layers', default=3, type=int)
        s_parser.add_argument('--hidden_sz', default=100, type=int)
        #opt and regularization
        s_parser.add_argument('--learning_rate', default=1e-3, type=float)
        s_parser.add_argument('--dropout', default=1.0, type=float)

        s_parser.add_argument('--allow_soft_placement', default=True, type=bool, help="Allow device soft device placement")
        s_parser.add_argument('--log_device_placement', default=False, type=bool, help="Log placement of ops on devices")

        #pdb.set_trace()

        args = parser.parse_args()

        #pdb.set_trace()
        batches,n_samples, vld_data = prepare_data(args)
        
        #pdb.set_trace()
        train(args, batches, n_samples, vld_data)

        pdb.set_trace()

    elif sys.argv[1]=='use_simulate_data': #use data generated by simulate_data

        s_parser = subs.add_parser('use_simulate_data')
         #IO
        s_parser.add_argument('--input_type', default=1, type=int) #use simulate_data data (sampled_seq_fa format and pairwise_distance format)
        s_parser.add_argument('--train_input1', type=str)
        s_parser.add_argument('--train_input2', type=str)
        s_parser.add_argument('--seq_type', type=int, help="0: binary 1: ATCG")
        s_parser.add_argument('--train_output_dir', type=str, help="dir path to store trained model")
        s_parser.add_argument('--max_num_to_sample', default=-1, type=int, help= "max number to sample from pairwise dist file for training data preparation")
        #train
        s_parser.add_argument('--batch_size', default=100, type=int)
        s_parser.add_argument('--num_epochs', default=500, type=int)
        s_parser.add_argument('--load_model', default=0, type=int, help="whether to load somewhat trained model (1) or not (0, default)")
        #model
        s_parser.add_argument('--maxlen', default=500, type=int, help="max len for input sequence")
        s_parser.add_argument('--blocklen', default=10, type=int, help="block len to segment input sequence")

        s_parser.add_argument('--embedding_size', default=200, type=int)
        s_parser.add_argument('--num_layers', default=3, type=int)
        s_parser.add_argument('--hidden_sz', default=100, type=int)
        #opt and regularization
        s_parser.add_argument('--learning_rate', default=1e-3, type=float)
        s_parser.add_argument('--dropout', default=1.0, type=float)

        s_parser.add_argument('--allow_soft_placement', default=True, type=bool, help="Allow device soft device placement")
        s_parser.add_argument('--log_device_placement', default=False, type=bool, help="Log placement of ops on devices")

        #pdb.set_trace()

        args = parser.parse_args()

        batches,n_samples, vld_data = prepare_data(args)

        train(args, batches, n_samples, vld_data)

    elif sys.argv[1]=='test_plot_log':

        s_parser = subs.add_parser('test_plot_log')
         #IO
        s_parser.add_argument('--input_log', type=str)
        s_parser.add_argument('--output_fig', type=str)
        
        args = parser.parse_args()

        plot_log(args.input_log, args.output_fig)
