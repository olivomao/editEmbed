'''
comprehensive data simulation
'''

from argparse import ArgumentParser
import pdb
import sys
import random
import numpy as np

from util import *

def random_dna_string(length):
    letters ="ATCG"
    return "".join(random.choice(letters) for i in range(length))

def random_bin_string(length):
    letters ="01"
    return "".join(random.choice(letters) for i in range(length))

def gen_cluster_center(args):

    logPrint("[gen_cluster_center]")

    pd, fn = parent_dir(args.output)
    run_cmd('mkdir -p %s'%pd)

    with open(args.output, 'w') as of:

        for i in range(args.num):

            #pdb.set_trace()

            sid = "%s%d"%(args.sid_pre, i)
            length = args.length

            if args.weight_distr == 0:
                weight = 1.0 / args.num
            else:
                weight = np.random.uniform(0,1)

            seq_description = ">%s\tgene=%s\tweight=%f"%(sid, sid, weight)
            
            if args.seq_type == 0:
                seq = random_bin_string(length)
            else:
                seq = random_dna_string(length)


            of.write("%s\n%s\n"%(seq_description, seq))

    logPrint("[gen_cluster_center] %s written"%args.output)

    return

'''
usage:
(1) generate cluster center in fasta format

    python simulate_data gen_cluster_center --output out_fasta_path (cluster_center_fa format)
                                            --seq_type 0 (Binary)/1 (ATCG)
                                            --num    num_clusters
                                            --length  seq_length
                                            [--weight_distr 0 (same)/1 (uniform)]
                                            [--sid_pre sid_prefix]


(2) sample cluster center
    (Later, we will sample cluster centers from real ATCG sequences)

(2) generate noisy samples in fasta format

    

(3) calculate distances

'''
if __name__ == "__main__":

    
    parser = ArgumentParser()
    subs = parser.add_subparsers()

    if sys.argv[1]=='gen_cluster_center':

        s_parser = subs.add_parser('gen_cluster_center')

        s_parser.add_argument('--output', type=str, help="output path of cluster center in cluster_center_fa format")
        s_parser.add_argument('--seq_type', type=int, help="type of sequence. 0: binary, 1: ATCG")
        s_parser.add_argument('--num', type=int, help="number of clusters")
        s_parser.add_argument('--length', type=int, help="length of sequence")
        s_parser.add_argument('--weight_distr', default=0, type=int, help="distribution to generate cluster centers. 0: same weight (default), 1: uniform")
        s_parser.add_argument('--sid_pre', default="c", type=str, help="sequence id prefix for output. Default is c e.g. c001 c002 etc")

        args = parser.parse_args()

        gen_cluster_center(args)

    else:

        pdb.set_trace()
