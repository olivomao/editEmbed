import pdb
#pdb.set_trace()
import tensorflow as tf 
import datetime
import time
from argparse import ArgumentParser
import numpy as np

#from global_vals import *
from proc_data import *
import inference
from tensorflow.python import debug as tf_debug

import matplotlib.pyplot as plt

'''
There're two source types:
(1) input_type==0: based on simulate_binary
    tain_input is in binary_seq_pair format.
(2) input_type==1: based on simulate_data
    train_input1: sampled_seq_fa format
    train_input2: pairwise_distance format
'''
def prepare_data(args):

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
    ax.plot(list_i, list_tr_loss, marker='o', label='training loss')
    ax.plot(list_i, list_vld_loss, marker='o', label='validation loss')
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

    plot_log(logPath, logFigPath)

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
