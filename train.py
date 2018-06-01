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
from model_enc_dec_2 import Model2
from logger import DevLogger

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
modified based on prepare_data_seq2seq

input: siamese_seq2seq_path (can be train or validation)
       seq_type, blocklen
output: s_i1_pah, s_i2_path, de_si, deviation_de_d_cgk, s_i_type

in particular, (si_1, si_2) seq pair in siamese_seq2seq_path split, with blocks of x into s_i1_path
                                                                    with blocks of x' into s_i2_path
               de_si and deviation_de_d_cgk will be dumped to de_si, deviation_de_d_cgk directly
               s_i_type indicates if (si_1, si_2) are independent or si_2 is derived by passing si_1
               through an indel channel (setting can be known from header of siamese_seq2seq_path)
'''
def prepare_data_siamese_seq2seq(siamese_seq2seq_path,
                                 seq_type,
                                 blocklen,
                                 s_i1_path,
                                 s_i2_path,
                                 de_si_path,
                                 deviation_de_d_cgk_path,
                                 s_i_type_path):
    #pdb.set_trace()

    with open(siamese_seq2seq_path, 'r') as in_f, \
         open(s_i1_path, 'w') as out_f1, \
         open(s_i2_path, 'w') as out_f2, \
         open(de_si_path, 'w') as out_de, \
         open(deviation_de_d_cgk_path, 'w') as out_dev, \
         open(s_i_type_path, 'w') as out_si_type:

        header =  '## src:%s\n'%siamese_seq2seq_path
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

            de = tokens[2]
            deviation_de_d_cgk = tokens[4]
            s_i_type = tokens[5]

            s_i1_blocks_str = get_seq2blocks(s_i1, seq_type, blocklen)
            s_i2_blocks_str = get_seq2blocks(s_i2, seq_type, blocklen)

            out_f1.write(s_i1_blocks_str+'\n')
            out_f2.write(s_i2_blocks_str+'\n')

            out_de.write(de+'\n')
            out_dev.write(deviation_de_d_cgk+'\n')
            out_si_type.write(s_i_type+'\n')

        logPrint('%s written'%s_i1_path)
        logPrint('%s written'%s_i2_path)
        logPrint('%s written'%de_si_path)
        logPrint('%s written'%deviation_de_d_cgk_path)
        logPrint('%s written'%s_i_type_path)

    #pdb.set_trace()
    return s_i1_path, s_i2_path, de_si_path, deviation_de_d_cgk_path, s_i_type_path


'''
header in input_log will be read and parsed into col_names
plot_itms is a list of group of col_names, shape=(n_subplots, n_curves). For example, plot_itms=[[a,b],[c,d]], a and b will be plotted in 1st subplot and c and d will be plotted in 2nd subplot
'''
def plot_log2(input_log, output_fig, plot_itms):

    logPrint('[plot_log2] starts')

    label_vals = {} #key: label from header val: list of vals
    label_colidx = {} #key: label val: col index in header
    colidx_label = {}
    with open(input_log,'r') as fi:
        header = fi.readline()
        header = header[1:] #skip # sign
        tokens = [t for t in header.split() if t!='']
        for i in range(len(tokens)):
            t = tokens[i]
            label_vals[t] = []
            label_colidx[t] = i
            colidx_label[i] = t

        #pdb.set_trace()
        for line in fi:
            if line[0]=='#': continue
            tokens = [convert_str_to_bool_int_float(t) for t in line.split() if t != '']
            if len(tokens)==0: continue
            for i in range(len(tokens)):
                label_vals[colidx_label[i]].append(tokens[i])
        #pdb.set_trace()

    n_subplots = len(plot_itms)

    fig, axes = plt.subplots(n_subplots)
    for i_plot in range(n_subplots):
        for label in plot_itms[i_plot]:
            x_vals = label_vals['i']
            y_vals = label_vals[label]
            axes[i_plot].plot(x_vals, y_vals, label=label)
            axes[i_plot].legend()

    #show/save all
    plt.tight_layout()
    plt.savefig(output_fig) #plt.show()
    #pdb.set_trace()

    logPrint('[plot_log2] ends. %s drawn'%output_fig)

    return

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
                               ("initializer", 
                                "source",
                                "target_input",
                                "target_output",
                                "source_sequence_length",
                                "target_sequence_length"))):
    pass

class SiameseSeq2Seq_BatchedInput(
        collections.namedtuple("BatchedInput",
                               ("initializer", 
                                "source",
                                "target_input",
                                "target_output",
                                "source_sequence_length",
                                "target_sequence_length",
                                "de_si",
                                "deviation_de_d_cgk",
                                "s_i_type"))):
    pass


'''
Based on:

- for seq2seq:
s_i1_path (x, in block chunks), s_i2_path (transformed_x, in block chunks) and vocab_path (including <unk> <s> and </s>), 
- for siamese seq2seq
additionally, we've de_si_path, and deviation_de_d_cgk_path, and s_i_type_path for metric evaluation purpose (we calculate them in advance to speedup run time efficienty)

We build a lookup table from vocab and then transform x and transformed_x from block chunks into ids.
In addition, [1]+tgt ==> tgt_input and tgt+[2] ==> tgt_output, and all cols padded (w/ 2, eos) to have same length across batch.

- for seq2seq:
output an iterator (src_ids, tgt_input, tgt_output, src_len/before padding, tgt_len/before padding) 
- for siamese seq2seq:
output an iterator (src_ids/s_i1, tgt_input/s_i2, tgt_output/dummy, src_len/before padding, tgt_len/before padding, de_si, deviation_de_d_cgk, s_i_type)

The iterator can be used for "train" or "infer" purpose

'''

# a wrapper of prepare_minibatch_seq2seq for code reuse
# ipt_files are of TfDatasetInputFiles
# purpose "train" use args.batch_size "infer" use args.n_cluster_validation

def prepare_minibatch_siamese_seq2seq(ipt_files, args, purpose):

    batched_s_i1  = prepare_minibatch_seq2seq(
                        ipt_files.s_i1_path,#only batched_s_i1.source and .source_length will be used
                        ipt_files.s_i1_path,#dummy
                        ipt_files.vocab_path,
                        args,
                        purpose,
                        ipt_files.de_si_path,
                        ipt_files.deviation_de_d_cgk_path,
                        ipt_files.s_i_type_path)

    batched_s_i2  = prepare_minibatch_seq2seq(
                        ipt_files.s_i2_path,#only batched_s_i2.source and .source_length will be used
                        ipt_files.s_i2_path,#dummy
                        ipt_files.vocab_path,
                        args,
                        purpose,
                        ipt_files.de_si_path,
                        ipt_files.deviation_de_d_cgk_path,
                        ipt_files.s_i_type_path)

    return batched_s_i1, batched_s_i2

def prepare_minibatch_seq2seq(s_i1_path, 
                              s_i2_path, 
                              vocab_path, 
                              args,
                              purpose,
                              de_si_path=None,#for siamese seq2seq
                              deviation_de_d_cgk_path=None,
                              s_i_type_path=None):
    
    '''
    open a textline dataset, remove comment and empty lines
    '''
    def open_dataset(path):
        ds = tf.data.TextLineDataset(path)
        ds = ds.filter(lambda line: \
                       tf.logical_and(tf.not_equal(tf.substr(line, 0, 1), "#"),
                       tf.not_equal(line, '')))
        return ds

    table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocab_path,
                                                    default_value=0, #<unk> defined in vocab_path
                                                   )
    
    src_dataset = open_dataset(s_i1_path)
    tgt_dataset = open_dataset(s_i2_path)

    if de_si_path is not None:
       de_si_dataset = open_dataset(de_si_path)
       de_si_dataset = de_si_dataset.map(lambda line: \
                                         tf.string_to_number(line, out_type=tf.float32))
       dev_dataset = open_dataset(deviation_de_d_cgk_path)
       dev_dataset = dev_dataset.map(lambda line: \
                                     tf.string_to_number(line, out_type=tf.float32))
       s_i_type_dataset = open_dataset(s_i_type_path)
       s_i_type_dataset = s_i_type_dataset.map(lambda line: \
                                               tf.string_to_number(line, out_type=tf.int32))
       src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, de_si_dataset, dev_dataset, s_i_type_dataset))
    else:
       src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(args.output_buffer_size, args.random_seed)
    src_tgt_dataset = src_tgt_dataset.repeat()

    if de_si_path is None:
        src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: \
                                              (tf.string_split([src]).values,
                                               tf.string_split([tgt]).values)).\
                                             prefetch(args.output_buffer_size)
        #words/blocks to ids
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: \
                                              (tf.cast(table.lookup(src), tf.int32),
                                               tf.cast(table.lookup(tgt), tf.int32))).\
                                             prefetch(args.output_buffer_size)

        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: \
                                              (src, 
                                               tf.concat(([1], tgt), 0),
                                               tf.concat((tgt, [2]), 0))).\
                                             prefetch(args.output_buffer_size) #1: start of sentence and 2: end of sentence
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out: \
                                              (src,
                                               tgt_in, 
                                               tgt_out, 
                                               tf.size(src), 
                                               tf.size(tgt_in))).prefetch(args.output_buffer_size)

    else:#siamese seq2seq        
        src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt,de,dv,st: \
                                              (tf.string_split([src]).values,
                                               tf.string_split([tgt]).values,
                                               de,
                                               dv,
                                               st)).\
                                             prefetch(args.output_buffer_size)
        #words/blocks to ids
        src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt,de,dv,st: \
                                              (tf.cast(table.lookup(src), tf.int32),
                                               tf.cast(table.lookup(tgt), tf.int32),
                                               de,
                                               dv,
                                               st)).\
                                             prefetch(args.output_buffer_size)

        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, de, dv, st: \
                                              (src, 
                                               tf.concat(([1], tgt), 0),
                                               tf.concat((tgt, [2]), 0),
                                               de,
                                               dv,
                                               st)).\
                                             prefetch(args.output_buffer_size) #1: start of sentence and 2: end of sentence
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out, de, dv, st: \
                                              (src,
                                               tgt_in, 
                                               tgt_out, 
                                               tf.size(src), 
                                               tf.size(tgt_in),
                                               de,
                                               dv,
                                               st)).prefetch(args.output_buffer_size)




    if purpose == "train":
        batch_size = args.batch_size
    elif purpose == "infer":
        #pdb.set_trace()
        batch_size = args.n_clusters_validation

    if de_si_path is None:
        src_tgt_dataset = src_tgt_dataset.padded_batch(batch_size,
                                                   padded_shapes=(tf.TensorShape([None]),
                                                                  tf.TensorShape([None]),
                                                                  tf.TensorShape([None]),
                                                                  tf.TensorShape([]),
                                                                  tf.TensorShape([])),
                                                   padding_values=(2,2,2,0,0)) #padding tgt to same lengths; additional dim added to all cols of src_tgt_dataset
    else:#siamese seq2seq
        src_tgt_dataset = src_tgt_dataset.padded_batch(batch_size,
                                                   padded_shapes=(tf.TensorShape([None]),
                                                                  tf.TensorShape([None]),
                                                                  tf.TensorShape([None]),
                                                                  tf.TensorShape([]),
                                                                  tf.TensorShape([]),
                                                                  tf.TensorShape([]),
                                                                  tf.TensorShape([]),
                                                                  tf.TensorShape([])),
                                                   padding_values=(2,2,2,0,0,0.0,0.0,0))

    iterator = src_tgt_dataset.make_initializable_iterator() #src_tgt_dataset.batch(args.batch_size).make_initializable_iterator()
    
    if de_si_path is None:
        (src_ids, \
         tgt_input_ids, \
         tgt_output_ids, \
         src_seq_len, \
         tgt_seq_len) = iterator.get_next()

    else: #siamese seq2seq
        (src_ids, \
         tgt_input_ids, \
         tgt_output_ids, \
         src_seq_len, \
         tgt_seq_len,\
         de,\
         dv,\
         st) = iterator.get_next()
 
    #remove outer dimension
    #src_ids = tf.squeeze(src_ids, axis=[0]); print(src_ids)
    #tgt_input_ids = tf.squeeze(tgt_input_ids, axis=[0]); #print(tgt_input_ids)
    #tgt_output_ids = tf.squeeze(tgt_output_ids, axis=[0]); #print(tgt_output_ids)
    #src_seq_len = tf.squeeze(src_seq_len, axis=[0]); #print(src_seq_len)
    #tgt_seq_len = tf.squeeze(tgt_seq_len, axis=[0]); #print(tgt_seq_len)
    #pdb.set_trace()

    if de_si_path is None:
        bi = BatchedInput(initializer=iterator.initializer,
                      source=src_ids,
                      target_input=tgt_input_ids,
                      target_output=tgt_output_ids,
                      source_sequence_length=src_seq_len,
                      target_sequence_length=tgt_seq_len)
    
    else: #siamese seq2seq
        
        bi = SiameseSeq2Seq_BatchedInput(initializer=iterator.initializer,
                      source=src_ids,
                      target_input=tgt_input_ids,
                      target_output=tgt_output_ids,
                      source_sequence_length=src_seq_len,
                      target_sequence_length=tgt_seq_len,
                      de_si=de,
                      deviation_de_d_cgk=dv,
                      s_i_type=st)

    return bi #BatchedInput

'''
input:
    generated_ids shape=(batch size, max_time_1)
    reference_ids shape=(batch size, max_time_2)
output:
    mean (averaged across batch size) hamming distance between per row of (generated_ids, reference_ids)
    mean of ref seq lengths
    mean of gen/predicted seq lengths

For example, per row of (generated_ids, reference_ids) are:
    ([3,4,5,6,6], [3,4,5,2]) here 1st list is not finished (stopped by '2' or </s>)
                             1st list translated to 00 01 10 11 11 and 2nd list translated to 00 01 10 (blocklen=2, 3==>'00' because 0,1,2 correspond to unk, <s>, </s>) and their hamming distance is 4 
'''
def calc_validation_loss(generated_ids, reference_ids, blocklen):

    def get_id(id):
        if id<3: #<unk> <s> </s>
            #print('id %d occurred'%id)
            return 0
        else:
            return id-3
                                            
    def id2bseq(id):
        return ("{0:0%db}"%blocklen).format(id)
                                                    
    def get_seq(list_ids):
        return ''.join([id2bseq(get_id(list_ids[i])) for i in range(len(list_ids)) if list_ids[i]!=2])
                                                            
    generated_ids_seqs = [get_seq(generated_ids[r]) for r in range(len(generated_ids))]
    #print(str(generated_ids_seqs))

    reference_ids_seqs = [get_seq(reference_ids[r]) for r in range(len(reference_ids))]
    #print(str(reference_ids_seqs))

    def diff(seq1, seq2):
        L1 = len(seq1)
        L2 = len(seq2)
        d = max(L1,L2)-min(L1,L2)
        d += sum(c1 != c2 for c1, c2 in zip(seq1[:min(L1,L2)],seq2[:min(L1,L2)]))
        return d

    #print(diff('10011','0001'))

    hamming_distances = [diff(generated_ids_seqs[r], reference_ids_seqs[r]) for r in range(len(generated_ids))]
    #print(str(hamming_distances))

    generated_seq_lengths = [len(generated_ids_seqs[r]) for r in range(len(generated_ids))]

    reference_seq_lengths = [len(reference_ids_seqs[r]) for r in range(len(generated_ids))]

    return np.asarray(hamming_distances), np.mean(hamming_distances), np.mean(generated_seq_lengths), np.mean(reference_seq_lengths)

# train_logits shape=(batch size, time, vocab)
def logits2ids(train_logits):

    #pdb.set_trace()
    train_ids = np.argmax(train_logits, axis=2)

    return train_ids


def check_variables(msg):
    vs= tf.trainable_variables()
    print('%s. There are %d trainable variables'%(msg, len(vs)))
    for v in vs:
        print(v)

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


    #========== prepare batched/shuffled data via tf.dataset
    batched_input   = prepare_minibatch_seq2seq(s_i1_path,
                                               s_i2_path,
                                               vocab_path,
                                               args,
                                               "train")

    batched_input_infer = prepare_minibatch_seq2seq(s_i1_path_vld,
                                                    s_i2_path_vld,
                                                    vocab_path,
                                                    args,
                                                    "infer")
    #pdb.set_trace()
    #def check_variables(msg):
    #    vs= tf.trainable_variables()
    #    print('%s. There are %d trainable variables'%(msg, len(vs)))
    #    for v in vs:
    #        print(v)

    #========== build model/ computational graph
    #pdb.set_trace()
    #model = Model(args, batched_input, batched_input_infer)
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
        model_train = Model2(args, 
                             batched_input,
                             purpose="train",
                             model_name="model_train")
        check_variables('After model_train created')
        model_infer = Model2(args,
                             batched_input_infer,
                             purpose="infer",
                             model_name="model_infer")
        check_variables('After model_infer created')
     
    #========== optimizer/ gradients operation
    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    #grads_and_vars = optimizer.compute_gradients(model.loss)
    grads_and_vars = optimizer.compute_gradients(model_train.loss)

    capped_gvs = []
    for grad, var in grads_and_vars:
        if grad is not None:
            #capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
            capped_gvs.append((tf.clip_by_norm(grad, args.grad_max), var))
        else:
            capped_gvs.append((grad, var))

    tr_op_set = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    #========== session run
    with tf.Session() as sess:

        #logging
        saver = tf.train.Saver()
        logPath = args.model_dir_path+'/log.txt'
        logFigPath = args.model_dir_path+'/log.png' 
        #'''
        logFile = open(logPath, 'w');

        logHeader = '#i\t'+\
                    'train_avg_crosent_loss\t'+\
                    'train_avg_hamming_loss\t'+\
                    'train_avg_predicted_len\t'+\
                    'train_avg_ref_len\t'+\
                    'validation_hamming_loss\t'+\
                    'validation_predicted_len\t'+\
                    'validation_ref_len\n'
        print(logHeader)
        logFile.write(logHeader)

        #actual train
        sess.run(tf.tables_initializer())
        sess.run(batched_input.initializer)
        sess.run(batched_input_infer.initializer)
        sess.run(tf.global_variables_initializer())

        #
        i_batch = 0
        period_to_dump_log = 5
        period_to_save_model = 100
        stat = {'train_avg_crosent_loss':[],
                'train_avg_hamming_loss':[],
                'train_avg_predicted_len':[],
                'train_avg_ref_len':[]}
        for ep in range(args.n_epoch):
            for b in range(args.n_clusters/args.batch_size):
                #pdb.set_trace()
                i_batch += 1
                #print('ep=%d,batch=%d'%(ep,b))
                
                #train data
                #_, batch_step, batch_loss,\
                #train_logits, train_tgt_output = \
                #    sess.run([tr_op_set, global_step, model.loss,
                #              model.logits,
                #              batched_input.target_output])
                _,\
                batch_step,\
                batch_loss,\
                train_logits,\
                train_tgt_output = \
                    sess.run([tr_op_set,
                              global_step, 
                              model_train.loss,
                              model_train.logits,
                              batched_input.target_output])

                train_ids = logits2ids(train_logits)

                #pdb.set_trace()
                _,\
                train_hamming_loss,\
                train_predicted_len,\
                train_ref_len = \
                    calc_validation_loss(train_ids,
                                         train_tgt_output,
                                         blocklen=args.blocklen)
                
                if np.isnan(batch_loss):
                    print('batch_loss (train crosent loss) nan')
                    logFile.close()
                    return

                stat['train_avg_crosent_loss'].append(batch_loss)
                stat['train_avg_hamming_loss'].append(train_hamming_loss)
                stat['train_avg_predicted_len'].append(train_predicted_len)
                stat['train_avg_ref_len'].append(train_ref_len)

                #validation/test data
                #if i_batch>50: pdb.set_trace()
                if i_batch % period_to_dump_log == 0:
                    #pdb.set_trace()
                    #infer_ids, infer_tgt_output = \
                    #    sess.run([model.sample_id_infer, \
                    #              batched_input_infer.target_output])
                    infer_ids, \
                    infer_tgt_output = \
                        sess.run([model_infer.sample_id, \
                                  batched_input_infer.target_output])
            

                    _,\
                    validation_hamming_loss,\
                    validation_predicted_length, \
                    validation_ref_length = \
                        calc_validation_loss(infer_ids, 
                                             infer_tgt_output,
                                             blocklen=args.blocklen)

                    if np.isnan(validation_hamming_loss):
                        print('validation hamming loss nan')
                        logFile.close()
                        return

                    logMsg = ""
                    logMsg += "%d\t"%i_batch
                    #train - cross entropy loss
                    logMsg += "%f\t"%np.mean(stat['train_avg_crosent_loss'])
                    #train - hamming loss
                    logMsg += "%f\t"%np.mean(stat['train_avg_hamming_loss'])
                    logMsg += "%f\t"%np.mean(stat['train_avg_predicted_len'])
                    logMsg += "%f\t"%np.mean(stat['train_avg_ref_len'])
                    for k in stat.keys():
                        stat[k]=[]

                    logMsg += "%f\t"%validation_hamming_loss
                    logMsg += "%f\t"%validation_predicted_length
                    logMsg += "%f\t"%validation_ref_length
                    print('ep=%d b=%d %s'%(ep, b, logMsg))

                    logFile.write(logMsg+'\n')

                if i_batch % period_to_save_model == 0:
                    try:
                        print("save model")
                        saver.save(sess, \
                            args.model_dir_path+\
                            'ckpt_step_%d_loss_%s'%(i_batch,
                            str(batch_loss)))
                        print(logHeader)
                    except:
                        print("saver.save exception")
                        if logFile.closed==False: logFile.close()
                        return

        #pdb.set_trace()
        if logFile.closed==False: logFile.close()
        #pdb.set_trace()
        #'''
        plot_log2(logPath, logFigPath,[['train_avg_crosent_loss',
                                        'train_avg_hamming_loss',
                                        'validation_hamming_loss'],
                                       ['train_avg_predicted_len',
                                        'train_avg_ref_len'],
                                       ['validation_predicted_len',
                                        'validation_ref_len']]) 
        print('TRAIN FINISH'); #pdb.set_trace()
    return

'''
SNA ED - siamese network architecture - encoder decoder

mainly, for grad update and performance evaluation purpose,
stores the state (s_i1, s_i2)/input pair of sequences block ids
                  per shape: (bs, time)
       and action (a_i1_ids, a_i2_ids)/output predicted pair of sequences block ids
                  per shape: (bs, time)
       and action logits (a_i1_logits, a_i2_logits)
                  per shape: (bs, time, vocab_size)

additionally, for performance evaluation purpose,
    de_si: shape of (bs,)
           edit distance of (s_i1, s_i2)
    dev_de_d_cgk: shape of (bs,)
                  d_H(cgk(s_i1), cgk(s_i2))/de_si - 1
    s_i_type: shape of (bs,) 
              0 means (s_i1, s_i2) independent and 1 dependent

'''
class SNA_ED_StateAction(
    collections.namedtuple("SNA_ED_StateAction",
                           ("s_i1_src", 
                            "a_i1_logits",
                            "a_i1_ids",
                            "s_i2_src",
                            "a_i2_logits",
                            "a_i2_ids",
                            "de_si",
                            "dev_de_d_cgk",
                            "s_i_type"))):
    pass

'''
input:
    batched_s_i1 with shape (bs, time)
    batched_s_i2 with shape (bs, time)
    into model_s_i1 and model_s_i2 respectively
output:
    SNA_ED_StateAction object
'''
def state2action(sess,
                 batched_s_i1,
                 batched_s_i2,
                 model_s_i1,
                 model_s_i2):
    s_i1_src,\
    a_i1_logits,\
    a_i1_ids,\
    de_si,\
    dev_de_d_cgk, \
    s_i_type     = sess.run([batched_s_i1.source,
                             model_s_i1.logits, #_infer,
                             model_s_i1.sample_id, #_infer,
                             batched_s_i1.de_si,
                             batched_s_i1.deviation_de_d_cgk,
                             batched_s_i1.s_i_type])


    s_i2_src,\
    a_i2_logits,\
    a_i2_ids     = sess.run([batched_s_i2.source,
                             model_s_i2.logits, #_infer,
                             model_s_i2.sample_id]) #_infer])

    res = SNA_ED_StateAction(s_i1_src=s_i1_src,
                             a_i1_logits=a_i1_logits,
                             a_i1_ids=a_i1_ids,
                             s_i2_src=s_i2_src,
                             a_i2_logits=a_i2_logits,
                             a_i2_ids=a_i2_ids,
                             de_si=de_si,
                             dev_de_d_cgk=dev_de_d_cgk,
                             s_i_type=s_i_type)

    return res

'''
Containing metrics for performance evaluation.
To be augmented in future

In particular,

dev_de_d_nn: d_H(nn(s_i1), nn(s_i2))/d_e(s_i1, s_i2)-1
dev_de_d_cgk: d_H(cgk(s_i1), cgk(s_i2))/d_e(s_i1, s_i2)-1
d_H_nn:  d_H(nn(s_i1), nn(s_i2))
d_H_cgk:  d_H(cgk(s_i1), cgk(s_i2))
'''
class Metrics(
    
    collections.namedtuple("Metrics",
                           ("dev_de_d_nn",
                            "dev_de_d_cgk",
                            "d_H_nn",
                            "d_H_cgk"
                           ))):
    pass
                  
'''
input: SNA_ED_StateAction object
       blocklen
output:
       Metrics object - to be used for performance operation
'''
def calc_metrics(sna_ed_sa,
                 blocklen):

    d_H,_,_,_ = calc_validation_loss(sna_ed_sa.a_i1_ids,
                                     sna_ed_sa.a_i2_ids,
                                     blocklen)

    dev_de_d_nn = d_H/sna_ed_sa.de_si-1
    #post process illegal vals
    nan_indice = np.isnan(dev_de_d_nn)
    dev_de_d_nn[nan_indice] = 0
    inf_indice = np.isinf(dev_de_d_nn)
    dev_de_d_nn[inf_indice] = 0

    d_H_nn = (dev_de_d_nn+1)*sna_ed_sa.de_si
    dev_de_d_cgk = sna_ed_sa.dev_de_d_cgk
    d_H_cgk = (dev_de_d_cgk+1)*sna_ed_sa.de_si

    res = Metrics(dev_de_d_nn=dev_de_d_nn,
                  dev_de_d_cgk=dev_de_d_cgk,
                  d_H_nn=d_H_nn,
                  d_H_cgk=d_H_cgk) 
                    
    return res

'''
modified based on train_seq2seq
'''
def train_siamese_seq2seq_bkp(param_dic):

    for k,v in param_dic.items():
        param_dic[k] = convert_str_to_bool_int_float(v)
    args = Namespace(**param_dic)

    #==========process data for tf data pipeline
    run_cmd('mkdir -p %s'%os.path.join(args.model_dir_path, 'data_processed'))

    vocab_path = os.path.join(args.model_dir_path, 'data_processed', 'vocab.txt')
    prepare_vocab(vocab_path, args.seq_type, args.blocklen)

    s_i1_path =  os.path.join(args.model_dir_path, 'data_processed', 's_i1.seq2ids')
    s_i2_path =  os.path.join(args.model_dir_path, 'data_processed', 's_i2.seq2ids')
    de_si_path  =  os.path.join(args.model_dir_path, 'data_processed', 'de_si.txt')
    deviation_de_d_cgk_path  =  os.path.join(args.model_dir_path, \
                                             'data_processed', \
                                             'deviation_de_d_cgk.txt')
    s_i_type_path = os.path.join(args.model_dir_path, \
                                 'data_processed', \
                                 's_i_type.txt')
    #pdb.set_trace()
    prepare_data_siamese_seq2seq(\
                                args.siamese_seq2seq,
                                args.seq_type,
                                args.blocklen,
                                s_i1_path,
                                s_i2_path,
                                de_si_path,
                                deviation_de_d_cgk_path,
                                s_i_type_path)

    s_i1_path_validation =  os.path.join(args.model_dir_path,\
                                        'data_processed', \
                                        's_i1_validation.seq2ids')
    s_i2_path_validation =  os.path.join(args.model_dir_path, \
                                        'data_processed', \
                                        's_i2_validation.seq2ids')
    de_si_path_validation  =  os.path.join(args.model_dir_path, \
                                           'data_processed', \
                                           'de_si_validation.txt')
    deviation_de_d_cgk_path_validation  =  os.path.join(args.model_dir_path, \
                                             'data_processed', \
                                             'deviation_de_d_cgk_validation.txt')
    s_i_type_path_validation = os.path.join(args.model_dir_path, \
                                 'data_processed', \
                                 's_i_type_validation.txt')
    #pdb.set_trace()
    prepare_data_siamese_seq2seq(\
                                args.siamese_seq2seq_validation,
                                args.seq_type,
                                args.blocklen,
                                s_i1_path_validation,
                                s_i2_path_validation,
                                de_si_path_validation,
                                deviation_de_d_cgk_path_validation,
                                s_i_type_path_validation)
    #pdb.set_trace()
    #========== prepare batched/shuffled data via tf.dataset
    batched_s_i1  = prepare_minibatch_seq2seq(s_i1_path,#only batched_s_i1.source and .source_length will be used
                                              s_i1_path,#dummy
                                              vocab_path,
                                              args,
                                              "train",#use args.batch_size
                                              de_si_path,
                                              deviation_de_d_cgk_path,
                                              s_i_type_path)
    batched_s_i2  = prepare_minibatch_seq2seq(s_i2_path,#only batched_s_i2.source and .source_length will be used
                                              s_i2_path,#dummy
                                              vocab_path,
                                              args,
                                              "train",#use args.batch_size
                                              de_si_path,
                                              deviation_de_d_cgk_path,
                                              s_i_type_path)

    batched_s_i1_vld =    prepare_minibatch_seq2seq(s_i1_path_validation,#only batched_s_i1_vld.source and .source_length will be used
                                                    s_i1_path_validation,#dummy
                                                    vocab_path,
                                                    args,
                                                    "infer",#use args.n_cluster_validation
                                                    de_si_path_validation,
                                                    deviation_de_d_cgk_path_validation,
                                                    s_i_type_path_validation)
    batched_s_i2_vld =    prepare_minibatch_seq2seq(s_i2_path_validation,#only batched_s_i2_vld.source and .source_length will be used
                                                    s_i2_path_validation,#dummy
                                                    vocab_path,
                                                    args,
                                                    "infer",#use args.n_cluster_validation
                                                    de_si_path_validation,
                                                    deviation_de_d_cgk_path_validation,
                                                    s_i_type_path_validation)

    #========== build model/ computational graph for siamese architecture
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:#scope set as '' to be consistent with ckpt loading
        
        model_s_i1 = Model2(args, 
                            batched_s_i1,
                            purpose="infer",
                            model_name='model_s_i1') #model_s_i1.logits and .sample_id will be used (for grad update)
        model_s_i2 = Model2(args,
                            batched_s_i2,
                            purpose="infer",
                            model_name='model_s_i2') #model_s_i2.logits and .sample_id will be used (for grad update)
        
        model_s_i1_vld = Model2(args,
                                batched_s_i1_vld,
                                purpose="infer",
                                model_name='model_s_i1_vld') #model_s_i1_vld.logits and .sample_id will be used (for validation)
        model_s_i2_vld = Model2(args,
                                batched_s_i2_vld,
                                purpose="infer",
                                model_name='model_s_i2_vld') #model_s_i2_vld.logits and .sample_id will be used (for validation)
        #========== check trainable variables
        check_variables('train siamese seq2seq - trainable variables') 
        pdb.set_trace()

        #========== training session
        with tf.Session() as sess:

            ########## logger 
            deviation_logger = DevLogger(args.deviation_logger_path)
            deviation_logger_vld = DevLogger(args.deviation_logger_vld_path)
            
            sess.run(tf.tables_initializer())
            sess.run(batched_s_i1.initializer)
            sess.run(batched_s_i2.initializer)
            sess.run(batched_s_i1_vld.initializer)
            sess.run(batched_s_i2_vld.initializer)
            sess.run(tf.global_variables_initializer())


            saver = tf.train.Saver()

            #loaded model needs to be consistentwith created model here
            #ckpt = '/data1/shunfu/editEmbed/data_sim_data_type_bin_seq2seq/L50_TR10K_VLD2K/train/seq2seq_LSTM/single_set/ckpt_step_500_loss_35.4505'
            ckpt = '/data1/shunfu/editEmbed/data_sim_data_type_bin_seq2seq/L50_TR10K_VLD2K_TryNewModel/train/seq2seq_BiLSTM_Attention_tune_blocklen_embed_size/blocklen_5_embed_size_100/ckpt_step_2000_loss_4.18251'
            
            from tensorflow.python.tools.inspect_checkpoint \
                    import print_tensors_in_checkpoint_file

            print_tensors_in_checkpoint_file(file_name=ckpt, tensor_name='', all_tensors=False, all_tensor_names=True)

            saver.restore(sess, ckpt)
            print('%s loaded'%ckpt)

            ########## iteration
            for ep in range(args.n_epoch):

                for b in range(args.n_clusters/args.batch_size):

                    print('ep=%d b=%d'%(ep,b))
                    #if b==1: pdb.set_trace()

                    #pdb.set_trace()
                    sna_ed_sa = state2action(sess,
                                             batched_s_i1,
                                             batched_s_i2,
                                             model_s_i1,
                                             model_s_i2)

                    metrics = calc_metrics(sna_ed_sa, args.blocklen)

                    deviation_logger.add_log(
                            batch_index=[b]*len(sna_ed_sa.s_i1_src),
                            dev_cgk=metrics.d_H_cgk, #dev_de_d_cgk,
                            dev_nn=metrics.d_H_nn, #dev_de_d_nn,
                            s_i_type=sna_ed_sa.s_i_type)

                print('plot dH distribution for 1-epoch train data')
                deviation_logger.close() #do for one epoch
                deviation_logger.plot()

                if True: #check validation
                    print('plot dH distribution for all validation data')
                    sna_ed_sa_vld = state2action(sess,
                                                 batched_s_i1_vld,
                                                 batched_s_i2_vld,
                                                 model_s_i1_vld,
                                                 model_s_i2_vld)

                    metrics_vld = calc_metrics(sna_ed_sa_vld, args.blocklen)
                    deviation_logger_vld.add_log(
                            batch_index=[b]*len(sna_ed_sa_vld.s_i1_src),
                            dev_cgk=metrics_vld.d_H_cgk, #dev_de_d_cgk,
                            dev_nn=metrics_vld.d_H_nn, #dev_de_d_nn,
                            s_i_type=sna_ed_sa_vld.s_i_type)

                    deviation_logger_vld.close()
                    deviation_logger_vld.plot()
                pdb.set_trace()

    return

'''
prepared file paths (at model_dir_path/data_processed)
that are to be used by tf.dataset pipeline
'''
class TfDatasetInputFiles(
        collections.namedtuple("TfDatasetInputFiles",
            ("vocab_path",#for both training and validation batches
             "s_i1_path",#for training batches
             "s_i2_path",
             "de_si_path", 
             "deviation_de_d_cgk_path",
             "s_i_type_path",
            )
            )):
    pass

def prepare_siamese_seq2seq_models(args,
                                   batched_s_i1,
                                   batched_s_i2,
                                   batched_s_i1_vld,
                                   batched_s_i2_vld):

    #========== build model/ computational graph for siamese architecture
    model_s_i1 = None
    model_s_i2 = None
    model_s_i1_vld = None
    model_s_i2_vld = None

    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:#scope set as '' to be consistent with ckpt loading
        
        model_s_i1 = Model2(args, 
                            batched_s_i1,
                            purpose="infer",
                            model_name='model_s_i1') #model_s_i1.logits and .sample_id will be used (for grad update)
        model_s_i2 = Model2(args,
                            batched_s_i2,
                            purpose="infer",
                            model_name='model_s_i2') #model_s_i2.logits and .sample_id will be used (for grad update)
        
        model_s_i1_vld = Model2(args,
                                batched_s_i1_vld,
                                purpose="infer",
                                model_name='model_s_i1_vld') #model_s_i1_vld.logits and .sample_id will be used (for validation)
        model_s_i2_vld = Model2(args,
                                batched_s_i2_vld,
                                purpose="infer",
                                model_name='model_s_i2_vld') #model_s_i2_vld.logits and .sample_id will be used (for validation)

    return model_s_i1, model_s_i2, model_s_i1_vld, model_s_i2_vld

def prepare_tf_dataset_input_files(siamese_seq2seq, #input data for training
                                   model_dir_path,
                                   seq_type,
                                   blocklen,
                                   isValidationData=False
                                   ):

    if isValidationData==False:
        vs = '' #empty validation str
    else:
        vs = '_validation' #this naming is specified in storage_structure.txt

    run_cmd('mkdir -p %s'%os.path.join(model_dir_path, 'data_processed'))

    vocab_path = os.path.join(model_dir_path, 'data_processed', 'vocab.txt')
    prepare_vocab(vocab_path, seq_type, blocklen)

    s_i1_path =  os.path.join(model_dir_path, 'data_processed', 's_i1%s.seq2ids'%vs)
    s_i2_path =  os.path.join(model_dir_path, 'data_processed', 's_i2%s.seq2ids'%vs)
    de_si_path  =  os.path.join(model_dir_path, 'data_processed', 'de_si%s.txt'%vs)
    deviation_de_d_cgk_path  =  os.path.join(model_dir_path, \
                                             'data_processed', \
                                             'deviation_de_d_cgk%s.txt'%vs)
    s_i_type_path = os.path.join(model_dir_path, \
                                 'data_processed', \
                                 's_i_type%s.txt'%vs)
    #pdb.set_trace()
    prepare_data_siamese_seq2seq(\
                                siamese_seq2seq,
                                seq_type,
                                blocklen,
                                s_i1_path,
                                s_i2_path,
                                de_si_path,
                                deviation_de_d_cgk_path,
                                s_i_type_path)

    return TfDatasetInputFiles(
             vocab_path=vocab_path,#for both training and validation batches
             s_i1_path=s_i1_path,#for training batches
             s_i2_path=s_i2_path,
             de_si_path=de_si_path, 
             deviation_de_d_cgk_path=deviation_de_d_cgk_path,
             s_i_type_path=s_i_type_path,
            ) 

'''
update==False:
    augment acc_avg_r_per_batch

update==True:
    take mean of acc_avg_r_per_batch,
    update u_i, s_i and model param theta_i in vs_dic
    
    reset acc_avg_r_per_batch

- acc_avg_r_per_batch: list of avg_r_per_batch across different batches
- beta: learning rate to update u_i, s_i and thus theta_i (i is index for variables)

return vs_dic and acc_avg_r_per_batch
'''
def PGPE_update_theta(vs_dic, 
                      sna_ed_sa, 
                      metrics, 
                      acc_avg_r_per_batch,
                      sess,
                      beta,
                      update=False):

    #print('PGPE_update')

    rewards = - np.abs(metrics.dev_de_d_nn) 
    avg_reward_per_batch = np.mean(rewards)
    
    if update==False:
        acc_avg_r_per_batch.append(avg_reward_per_batch)
        return vs_dic, acc_avg_r_per_batch

    else:
        #pdb.set_trace()

        acc_avg_r_per_batch.append(avg_reward_per_batch)
        avg_r = np.mean(acc_avg_r_per_batch)
        acc_avg_r_per_batch = []

        for v_i in vs_dic.keys():
            u = vs_dic[v_i]['u']
            s = vs_dic[v_i]['s']

            v = v_i.eval() 

            d_u = (v - u)/np.square(s)
            ## for numerical stability
            nan_indice = np.isnan(d_u)
            d_u[nan_indice] = 0.0
            inf_indice = np.isinf(d_u)
            d_u[inf_indice] = 0.0

            D_u = avg_r * d_u
            u = u + beta*D_u
            vs_dic[v_i]['u']=u

            d_s = (np.square(v-u)-np.square(s))/(s*np.square(s))
            ## for numerical stability
            nan_indice = np.isnan(d_s)
            d_s[nan_indice] = 0.0
            inf_indice = np.isinf(d_s)
            d_s[inf_indice] = 0.0

            D_s = avg_r * d_s
            s = s + beta*D_s
            #s = max(0,s) #? negative deviation issue
            #s[s<0]=0
            pdb.set_trace()
            s = np.abs(s)
            vs_dic[v_i]['s']=s

            new_v = np.random.normal(loc=vs_dic[v_i]['u'], scale=vs_dic[v_i]['s'])
            v_i.load(new_v, sess)
        
    #pdb.set_trace()

    return vs_dic, acc_avg_r_per_batch


'''
rand init weights into the model
'''
def PGPE_rand_init(sess):

    print('PGPE_rand_init')

    vs_dic = {} #key: model trainable v, and val:{u:u_v, s:sigma_v}

    #pdb.set_trace()

    for v in tf.trainable_variables():
        vs_dic[v] = {}
        sp = v.eval().shape
        #vs_dic[v]['u']=np.random.uniform(-1.0, 1.0, size=sp) #u_v, v is with shape (?,?)
        vs_dic[v]['u']=v.eval()
        vs_dic[v]['s']=np.random.uniform(0, 0.01, size=sp) #sigma_v
        #for theta update purpose
        #vs_dic[v]['acc_r']=0.0
        #vs_dic[v]['acc_counter']=0

        new_v = np.random.normal(loc=vs_dic[v]['u'], scale=vs_dic[v]['s'])
        
        '''
        print(v)
        print(v.eval())

        print('u')
        print(vs_dic[v]['u'])

        print('s')
        print(vs_dic[v]['s'])

        print('new_v')
        print(new_v)
        '''

        v.load(new_v, sess)
        '''
        print('v after loading new_v')
        print(v.eval())
        #pdb.set_trace()
        '''
    return vs_dic

'''
initialize the model depending on RL_method and init_model_option

if RL_method==PGPE and init_model_option == 1,
  vc_dic is returned with key=i-th Variable theta(i) ~ N(u(i), sigma(i))
                          val=[u(i), sigma(i)] 
'''
def model_initialization(RL_method,
                         init_model_option,
                         sess,
                         saver):

    if RL_method == 0:#simply load or rand init a model
        return None
    elif RL_method == 1:#REINFORCE algo
        return None
    elif RL_method == 2:#PGPE algo
        if init_model_option == 0:#load a pretrained model
            #pdb.set_trace()
            load_pretrained_seq2seq(sess, saver)
            return None
        elif init_model_option == 1:#pre-load thea, and rand initialization theta ~ N(u=preload theta, sigma ~ U[0, 0.001])
            load_pretrained_seq2seq(sess, saver)
            vs_dic = PGPE_rand_init(sess)
            return vs_dic

'''
use saver to load a pretrained seq2seq model to sess

the pretrained model and current sess model should be compatible
'''
def load_pretrained_seq2seq(sess,
                            saver):

    #loaded model needs to be consistentwith created model here

    #ckpt = '/data1/shunfu/editEmbed/data_sim_data_type_bin_seq2seq/L50_TR10K_VLD2K/train/seq2seq_LSTM/single_set/ckpt_step_500_loss_35.4505'
    ckpt = '/data1/shunfu/editEmbed/data_sim_data_type_bin_seq2seq/L50_TR10K_VLD2K_TryNewModel/train/seq2seq_BiLSTM_Attention_tune_blocklen_embed_size/blocklen_5_embed_size_100/ckpt_step_2000_loss_4.18251'
        
    from tensorflow.python.tools.inspect_checkpoint \
            import print_tensors_in_checkpoint_file

    print_tensors_in_checkpoint_file(\
            file_name=ckpt, \
            tensor_name='', \
            all_tensors=False, \
            all_tensor_names=True)

    saver.restore(sess, ckpt)
    print('%s loaded'%ckpt)
    return

'''
modified to:
    - incorporate RL (e.g. REINFORCE or PGPE)
    - make blocks more modular
'''
def train_siamese_seq2seq(param_dic):

    #param_dic to args
    for k,v in param_dic.items():
        param_dic[k] = convert_str_to_bool_int_float(v)
    args = Namespace(**param_dic)

    #==========process data for tf data pipeline
    ipt_files = prepare_tf_dataset_input_files(
                                   args.siamese_seq2seq, #input data for training
                                   args.model_dir_path,
                                   args.seq_type,
                                   args.blocklen,
                                   isValidationData=False
                                   )

    ipt_files_vld = prepare_tf_dataset_input_files(
                                   args.siamese_seq2seq_validation, #input data for training
                                   args.model_dir_path,
                                   args.seq_type,
                                   args.blocklen,
                                   isValidationData=True
                                   )

    #pdb.set_trace()
    #========== prepare batched/shuffled data via tf.dataset
    #pdb.set_trace()
    batched_s_i1, \
    batched_s_i2 = prepare_minibatch_siamese_seq2seq(ipt_files, args, "train")

    batched_s_i1_vld, \
    batched_s_i2_vld = prepare_minibatch_siamese_seq2seq(ipt_files_vld, args, "infer")

    model_s_i1, model_s_i2,\
    model_s_i1_vld, model_s_i2_vld = \
    prepare_siamese_seq2seq_models(args,
                                   batched_s_i1,
                                   batched_s_i2,
                                   batched_s_i1_vld,
                                   batched_s_i2_vld)

    #========== check trainable variables
    check_variables('train siamese seq2seq - trainable variables') 
    #pdb.set_trace()

    #========== training session
    with tf.Session() as sess:

        ########## logger 
        deviation_logger = DevLogger(args.deviation_logger_path)
        deviation_logger_vld = DevLogger(args.deviation_logger_vld_path)

        ########## saver
        saver = tf.train.Saver()

        ########## initialization
        sess.run(tf.tables_initializer())
        sess.run(batched_s_i1.initializer)
        sess.run(batched_s_i2.initializer)
        sess.run(batched_s_i1_vld.initializer)
        sess.run(batched_s_i2_vld.initializer)
        sess.run(tf.global_variables_initializer())

        #if RL_method==2(PGPE) and init_model_option==1(pre-load+rand init)
        #    vs_dic is {theta(t):[u(t), sigma(t)]}
        #otherwise,
        #    vs_dic is None
        vs_dic = model_initialization(args.RL_method,
                             args.init_model_option,
                             sess,
                             saver)
        pdb.set_trace()

        if args.RL_method==2: #PGPE
            acc_avg_r_per_batch = []
            beta = args.beta
            PGPE_update_interval = 10 #number of batches after which to update policy
            b_cnt = 0
        #pdb.set_trace()
                 
        ########## iteration
        for ep in range(args.n_epoch):

            for b in range(args.n_clusters/args.batch_size):

                #pdb.set_trace()
                sna_ed_sa = state2action(sess,
                                         batched_s_i1,
                                         batched_s_i2,
                                         model_s_i1,
                                         model_s_i2)

                metrics = calc_metrics(sna_ed_sa, args.blocklen)
                
                print('ep=%d b=%d mean dev cgk=%f, nn=%f'%(ep,b,
                       np.mean(metrics.dev_de_d_cgk),
                       np.mean(metrics.dev_de_d_nn)))
                
                if args.RL_method==2: #PGPE
                    #pdb.set_trace()

                    b_cnt += 1
                    if b_cnt % PGPE_update_interval == 0:
                        update = True
                        b_cnt = 0
                    else:
                        update = False

                    if update==True:
                        print('current b_cnt=%d update'%b_cnt)

                    #pdb.set_trace()
                    vs_dic,\
                    acc_avg_r_per_batch = PGPE_update_theta(vs_dic, 
                                               sna_ed_sa,
                                               metrics,
                                               acc_avg_r_per_batch,
                                               sess,
                                               beta,
                                               update) 

                #deviation_logger.add_log(
                #        batch_index=[b]*len(sna_ed_sa.s_i1_src),
                #        dev_cgk=metrics.d_H_cgk, #dev_de_d_cgk,
                #        dev_nn=metrics.d_H_nn, #dev_de_d_nn,
                #        s_i_type=sna_ed_sa.s_i_type)

            #print('plot dH distribution for 1-epoch train data')
            #deviation_logger.close() #do for one epoch
            #deviation_logger.plot()

            if False: #check validation
                print('plot dH distribution for all validation data')
                sna_ed_sa_vld = state2action(sess,
                                             batched_s_i1_vld,
                                             batched_s_i2_vld,
                                             model_s_i1_vld,
                                             model_s_i2_vld)

                metrics_vld = calc_metrics(sna_ed_sa_vld, args.blocklen)
                deviation_logger_vld.add_log(
                        batch_index=[b]*len(sna_ed_sa_vld.s_i1_src),
                        dev_cgk=metrics_vld.d_H_cgk, #dev_de_d_cgk,
                        dev_nn=metrics_vld.d_H_nn, #dev_de_d_nn,
                        s_i_type=sna_ed_sa_vld.s_i_type)

                deviation_logger_vld.close()
                deviation_logger_vld.plot()
                pdb.set_trace()

    return


'''
show in terminal hist of dev_de_d_cgk and dev_de_d_nn
'''
def show_hist(n_bins, dev_de_d_cgk, dev_de_d_nn):
    
    a = dev_de_d_cgk
    b = dev_de_d_nn

    r_min = min(min(a), min(b))
    r_max = max(max(a), max(b))

    r = np.arange(r_min, r_max, (r_max-r_min)/n_bins)

    a_h, a_v = np.histogram(a, bins=r)
    #print(a_h, a_v)

    b_h, b_v = np.histogram(b, bins=r)
    #print(b_h, b_v)

    msg = ''.join('\t'.join(['%.3f'%r[i], str(a_h[i]), str(b_h[i])])+'\n' for i in range(len(r)-1))
    print('bin\tdev_de_d_cgk\tdev_de_d_nn\n'+msg)

    #print(r_min)
    #print(r_max)
    #print(r)
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
