import tensorflow as tf 
import datetime
import time
import pdb
from argparse import ArgumentParser

#from global_vals import *
from proc_data import *
import inference

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

    batches = batch_iter(x1, x2, y, args)

    n_samples = len(x1)

    return batches,n_samples


'''
input:
FLAGS      contains TF parameters for training; same as args (we use args)
batches    to yield batch data for training
n_samples: # of training samples

output:
model stored at FLAGS.ckpt_prefix
'''
def train(FLAGS,
          batches,
          n_samples
          ):

    #load model
    siamese = inference.siamese(FLAGS)

    #optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    grads_and_vars = optimizer.compute_gradients(siamese.loss)

    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    #session
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    #
    saver = tf.train.Saver()

    #actual train
    sess.run(tf.global_variables_initializer())

    if FLAGS.load_model==True:
        saver.restore(sess, FLAGS.ckpt_prefix)
    #pdb.set_trace()
    n_batches = n_samples/FLAGS.batch_size * FLAGS.num_epochs

    for i in range(n_batches):

        b_x1, b_x2, b_y = batches.next()

        _, b_step, b_loss = sess.run([tr_op_set, global_step, siamese.loss], feed_dict={siamese.input_x1: b_x1,
                                                                                        siamese.input_x2: b_x2,
                                                                                        siamese.input_y:  b_y, 
                                                                                        siamese.dropout:  FLAGS.dropout})

        if b_step % 100 == 0:
            try:
                print("train step=%d, loss=%s"%(b_step, str(b_loss)))
                if b_step % 1000 == 0:
                    #pdb.set_trace()
                    print("save model b_step=%d"%b_step)
                    saver.save(sess, FLAGS.train_output_dir+'ckpt')
            except:
                print("train step=%d exception"%(b_step))
                break

    #pdb.set_trace()

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
        batches,n_samples = prepare_data(args)
        
        #pdb.set_trace()
        train(args, batches, n_samples)

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

        batches,n_samples = prepare_data(args)

        train(args, batches, n_samples)