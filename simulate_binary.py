from argparse import ArgumentParser
import pdb
import sys
import random
import numpy as np

from indel_channel import *
from edit_dist import *

#simulate binary sequences

def random_binary_string(length):
    letters ="01"
    return "".join(random.choice(letters) for i in range(length))

def simulate_binary(args):

    of = open(args.o, "w")
    python_cmd = " ".join(sys.argv)
    of.write("## %s\n"%python_cmd)

    for i in range(args.N):

        x_seq = random_binary_string(args.L)
        y_seq = sim_seq_binary(x_seq, args.s, args.d, args.i)
        dist = edit_dist(x_seq, y_seq)

        #of.write("[%d]\n"%(i+1))
        #of.write("%s\n"%x_seq)
        #of.write("%s\n"%y_seq)
        #of.write("%d\n"%dist)

        of.write("%s\t"%x_seq)
        of.write("%s\t"%y_seq)
        of.write("%d\n"%dist)

    of.close()

    return

'''
python simulate_binary.py -N num_seq
                          -L seq_len
                          [-i insertion_rate]
                          [-d deletion_rate]
                          [-s substitution_rate]
                          [-D distance_metric]
                          -o output_file
note:
1. output_file in binary_seq_pair format
'''
if __name__ == "__main__":

    parser = ArgumentParser("simulate binary sequences")
    parser.add_argument("-N", type=int, help="Number of sequence pairs")
    parser.add_argument("-L", type=int, help="Seq length of x")
    parser.add_argument("-i", type=float, default=0, help="Insert error rate [0-1] (default: 0)")
    parser.add_argument("-d", type=float, default=0, help="Deletion error rate [0-1] (default: 0)")
    parser.add_argument("-s", type=float, default=0, help="Substitution error rate [0-1] (default: 0)")
    parser.add_argument("-D", type=int, default=0, help="Type of distance_metric (default: 0 edit distance)")
    parser.add_argument("-o", type=str, help="Output path in binary_seq_pair format")

    args = parser.parse_args()

    simulate_binary(args)
