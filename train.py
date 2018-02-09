import tensorflow as tf 
import datetime
import time
import pdb

from global_vals import *
from proc_data import *
import inference

'''
train model
'''
def train():

    #params
    #ckpt

    case_dir = 'data/case_%s/'%str(int(time.time())) #datetime.datetime.now().isoformat()
    tf.flags.DEFINE_boolean("load_model", False, "Load existing model")
    tf.flags.DEFINE_string("ckpt_prefix", case_dir+'ckpt', "path of the check point data")

    tf.flags.DEFINE_string("train_input", "data/train.txt", "path of the training input in binary_seq_pair format")
    tf.flags.DEFINE_integer("embedding_size", 200, "dimension of block seq (of BLOCKLEN length) embedding")
    tf.flags.DEFINE_integer("num_layers", 3, "Number of hidden layers per siamese side")
    tf.flags.DEFINE_integer("hidden_sz", 100, "Size of hidden units" )
    tf.flags.DEFINE_integer("batch_size", 100, "Batch size" ) #batch_size
    tf.flags.DEFINE_integer("num_epochs", 500, "Number of epochs" ) #num_epochs
    #Misc Parameters (related to HW deployment)
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


    FLAGS = tf.flags.FLAGS 

    FLAGS._parse_flags()

    #load training data

    x1, x2, y, _, _ = load(FLAGS.train_input)

    batches = batch_iter(x1, x2, y, FLAGS)

    pdb.set_trace()

    #load model
    siamese = inference.siamese(FLAGS)

    #optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimizer = tf.train.AdamOptimizer(1e-3)

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

    n_batches = len(x1)/FLAGS.batch_size * FLAGS.num_epochs

    for i in range(n_batches):

        b_x1, b_x2, b_y = batches.next()

        _, b_step, b_loss = sess.run([tr_op_set, global_step, siamese.loss], feed_dict={siamese.input_x1: b_x1,
                                                                                        siamese.input_x2: b_x2,
                                                                                        siamese.input_y:  b_y, 
                                                                                        siamese.dropout:  1.0})

        if b_step % 100 == 0:
            try:
                print("train step=%d, loss=%d"%(b_step, b_loss))
                if b_step % 1000 == 0:
                    print("save model b_step=%d"%b_step)
                    saver.save(sess, FLAGS.ckpt_prefix)
            except:
                print("train step=%d exception"%(b_step))
                break

    #pdb.set_trace()

    return

if __name__ == "__main__":

    train()