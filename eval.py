import tensorflow as tf 
import pdb
import numpy as np 

from proc_data import *
import inference

def config():

    tf.flags.DEFINE_string("case_dir", "/data1/shunfu1/tf_tutorial/editEmbedding/data/case_2/", "path where ckpt & eval results are stored")

    tf.flags.DEFINE_string("eval_input", "/data1/shunfu1/tf_tutorial/editEmbedding/data/train.txt", "file to be evaluated")

    tf.flags.DEFINE_string("output", "/data1/shunfu1/tf_tutorial/editEmbedding/data/case_2/res.txt", "res file")


    #init model
    tf.flags.DEFINE_integer("embedding_size", 200, "dimension of block seq (of BLOCKLEN length) embedding")
    tf.flags.DEFINE_integer("num_layers", 2, "Number of hidden layers per siamese side")
    tf.flags.DEFINE_integer("hidden_sz", 50, "Size of hidden units" )

    #misc

    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


    FLAGS = tf.flags.FLAGS 

    FLAGS._parse_flags()

    return FLAGS

def restore_sess(FLAGS):

    #pdb.set_trace()

    ckpt = FLAGS.case_dir + "ckpt" #can't add '/' before 'ckpt'

    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)

    saver = tf.train.import_meta_graph("%s.meta"%ckpt) #tf.train.Saver()

    sess.run(tf.global_variables_initializer())    

    saver.restore(sess, ckpt)

    return sess

def draw_histogram(val_list, FLAGS):

    import matplotlib.pyplot as plt 

    plt.hist(val_list, normed=True, bins=30)

    fig_path = '%s/fig.png'%FLAGS.case_dir

    plt.savefig(fig_path) #plt.show()

    print('fig saved at %s'%fig_path)

    #pdb.set_trace()

    return

def eval():

    FLAGS = config()

    x1, x2, y, x1_str, x2_str = load(FLAGS.eval_input)
    
    batches = batch_eval_iter(x1, x2, y)

    #'''
    #session
    graph = tf.Graph()

    with graph.as_default():

        sess = restore_sess(FLAGS)

        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]

        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]

        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout = graph.get_operation_by_name("dropout").outputs[0]

        # Tensors we want to evaluate
        distance = graph.get_operation_by_name("distance/distance").outputs[0]

        loss = graph.get_operation_by_name("loss/loss").outputs[0]

        diff_list = []

        with open(FLAGS.output, 'w') as of:

            for i in range(len(x1)):

                b_x1, b_x2, b_y = batches.next()

                #pdb.set_trace()

                distance_i, loss_i = sess.run([distance, loss], feed_dict={   input_x1: b_x1,
                                                                              input_x2: b_x2,
                                                                              input_y:  b_y,
                                                                              dropout:  1.0})
                if y[i] != 0:
                    diff = float(distance_i-y[i])/y[i]
                else:
                    continue
                of.write("x1=%s\tx2=%s\ty=%d\test_y=%d\tdiff=%f\n"%(x1_str[i], x2_str[i], y[i], distance_i, diff))

                diff_list.append(diff)
                #pdb.set_trace()

            print("%s written"%FLAGS.output)

            draw_histogram(diff_list, FLAGS)

    return

if __name__ == "__main__":

    eval()