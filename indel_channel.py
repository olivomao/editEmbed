#!/usr/bin/env python

#modified from cDNA_Cupcake/simulate/simulate.py

__version__ = '1.0'

import os, sys, pdb
import random
from collections import namedtuple, defaultdict
from Bio import SeqIO

from util import *

simType = ['sub', 'ins', 'del', 'match']
simTypeSize = 4

def throwdice(profile):
    dice = random.random()
    for i in xrange(simTypeSize):
        if dice < profile[i]:
            return simType[i]
        
def sim_start(ntimes, profile):
    start = defaultdict(lambda: 0)
    acc = defaultdict(lambda: 0)
    for i in xrange(ntimes):
        curpos = 0
        while True:
            type = throwdice(profile)
            acc[type] += 1
            if type=='match': 
                start[curpos] += 1
                break
            elif type=='ins':
                # insertion occurred, with 1/4 chance it'll match
                if random.random() < .25:
                    start[curpos] += 1
                    break
            # if is sub or del, just advance cursor forward
            curpos += 1
    return start, acc

def sim_seq(seq, profile):
    nucl = set(['A','T','C','G'])
    sim = ''
    for i, s in enumerate(seq):
        while True:
            type = throwdice(profile)
            if type=='match': 
                sim += s
                break
            elif type=='ins':
                # insertion occurred, with 1/4 chance it'll match
                choice = random.sample(nucl,1)[0]
                sim += choice
            elif type=='sub': # anything but the right one
                choice = random.sample(nucl.difference([s]),1)[0]
                sim += choice
                break
            elif type=='del': # skip over this
                break
            else: raise KeyError, "Invalid type {0}".format(type)
        
    return sim


def sim_seq_dna(seq, sub, dele, ins):

    if(sub+ins+dele>=1):
        print >> sys.stderr, "Total sub+ins+del error can not exceed 1!"
        sys.exit(-1)

    profile = [sub, sub+ins, sub+ins+dele, 1.]

    return sim_seq(seq, profile)
 
    

'''
simulate a binary seq through a long read channel

input:
seq - input binary seq
sub - sub rate
ins - ins rate
dele - dele rate
(need: 0 <= sub + ins + del < 1)
'''
def sim_seq_binary(seq, sub, dele, ins):

    if(sub+ins+dele>=1):
        print >> sys.stderr, "Total sub+ins+del error can not exceed 1!"
        sys.exit(-1)

    profile = [sub, sub+ins, sub+ins+dele, 1.]

    return sim_seq_binary_profile(seq, profile)
    

def sim_seq_binary_profile(seq, profile):

    nucl = set(['0', '1'])
    sim = ''
    for i, s in enumerate(seq):
        while True:
            tp = throwdice(profile)
            if tp=='match': 
                sim += s
                break
            elif tp=='ins':
                # insertion occurred, with 1/4 chance it'll match
                choice = random.sample(nucl,1)[0]
                sim += choice
            elif tp=='sub': # anything but the right one
                choice = random.sample(nucl.difference([s]),1)[0]
                sim += choice
                break
            elif tp=='del': # skip over this
                break
            else: raise KeyError, "Invalid type {0}".format(tp)

    return sim

'''
input: fasta_filename in cluster_center_fa format

output: sampled sequences from cluster centers in sampled_seq_fa format
>seq_id     tid=cluster_seq_id      gene=gene_id    weight=wt
here, seq_id is [output_prefix]_[ith]

'''
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Simple error simulation")
    parser.add_argument("fasta_filename")
    parser.add_argument("output_prefix")
    parser.add_argument("--total_samples", type=int, default=-1, help="Total number of sampled sequences. Default: -1 "+\
                          "sample per cluster by --copy; >0, sample per cluster by its relative weight")
    parser.add_argument("--copy", type=int, default=1, help="Only used if --total_samples -1 "+\
                          "; Number of copies to simulate per cluster (default: 1)")        
    #parser.add_argument("--copy", type=int, default=1, help="Number of copies to simulate per input sequence (default: 1)")
    parser.add_argument("--ins", "-i", type=float, default=0, help="Insert error rate [0-1] (default: 0)")
    parser.add_argument("--dele", "-d", type=float, default=0, help="Deletion error rate [0-1] (default: 0)")
    parser.add_argument("--sub", "-s", type=float, default=0, help="Substitution error rate [0-1] (default: 0)")
    parser.add_argument("--output", "-o", type=str, help="seq output, in sampled_seq_fa format")
    parser.add_argument("--type", type=int, default=0, help="type of seq. 0 is binary; 1 is dna e.g. ATCG")

    args = parser.parse_args()

    if args.sub < 0 or args.sub > 1: 
        print >> sys.stderr, "Substitution error must be between 0-1!"
        sys.exit(-1)
    if args.ins < 0 or args.ins > 1:
        print >> sys.stderr, "Insertion error must be between 0-1!"
        sys.exit(-1)
    if args.dele < 0 or args.dele > 1:
        print >> sys.stderr, "Deletion error must be between 0-1!"
        sys.exit(-1)

    if args.sub + args.ins + args.dele > 1:
        print >> sys.stderr, "Total sub+ins+del error cannot exceed 1!"
        sys.exit(-1)


    #profile = [args.sub, args.sub+args.ins, args.ins+args.dele, 1.]
    profile = [args.sub, args.sub+args.ins, args.sub+args.ins+args.dele, 1.]

    fasta_filename = args.fasta_filename
    idpre = args.output_prefix

    seq_tp = args.type

    fo = open(args.output, 'w')

    ith = 0
    for r in SeqIO.parse(open(fasta_filename), 'fasta'):
        if args.total_samples==-1:
            n_copy = args.copy 
        else:
            #pdb.set_trace()
            n_copy = int(args.total_samples * float(r.description.strip().split()[-1][7:]))
            if n_copy==0:
                pdb.set_trace()
                print("%s has n_copy 0"%r.description)
                continue

        for j in xrange(n_copy): #xrange(args.copy):
            ith += 1
            #pdb.set_trace()
            #print(">{0}_{1}_{2}\n{3}".format(idpre, ith, r.id[:r.id.find('|')], sim_seq(r.seq.tostring(), profile)))
            #pdb.set_trace()
            #print(">{0}_{1} {2}\n{3}".format(idpre, ith, r.description, sim_seq(r.seq.tostring(), profile)))
            if seq_tp == 0:
                seq = sim_seq_binary_profile(str(r.seq), profile)
            else:
                seq = sim_seq(str(r.seq), profile)

            #pdb.set_trace()

            fo.write(">{0}_{1}\t{2}\n{3}\n".format(idpre, ith, "tid="+r.description, seq))

    fo.close()

    logPrint("[indel_channel] %s written"%args.output)  
