from argparse import ArgumentParser
import pdb, sys
import numpy as np

import matplotlib #draw_hist issue
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from util import logPrint, iterCounter
from inference import Predict, seq2nn


'''
input:
- pairwise_dist_file

output:

output data structures to be used for downstream analysis (e.g. histogram/ ROC)

- pairwise_dist_dic
  key: (pair_type, dist_col_idx) val: list of dist vals
  
  note: it's possible dist val is nan, in this case, the line will be skipped.
        number of such lines will be notified.   

- dist_label_dic
  key: dist_col_idx val: description of dist metric (if n/a, just str of key)
'''
def load_pairwise_dist(pairwise_dist_file):

    logPrint('[load_pairwise_dist] starts')

    #pdb.set_trace()

    dist_label_dic = {}

    with open(pairwise_dist_file, 'r') as fin:

        line = fin.readline()

        if line[0]=='#':

            tokens = line[1:].strip().split()

        else:

            tokens = line.strip().split()

        n_dist_metric = len(tokens) - 3

        for i in range(n_dist_metric):

            if line[0]=='#':

                dist_label_dic[i] = tokens[i+3]

            else:

                dist_label_dic[i] = str(i)

    #pdb.set_trace()

    pairwise_dist_dic = {}

    n_lines = sum([1 for line in open(pairwise_dist_file, 'r')])

    iterCnt = iterCounter(n_lines, 'load_pairwise_dist')

    #pdb.set_trace()

    cnt_nan = 0

    with open(pairwise_dist_file, 'r') as fin:

        for line in fin:

            iterCnt.inc()

            if line[0]=='#': continue

            tokens = line.strip().split()

            tp = int(tokens[2])

            has_nan = False 

            for i in range(len(tokens)-3):
                if tokens[i+3]=='nan':
                    #pdb.set_trace()
                    has_nan = True
                    break
            if has_nan:
                cnt_nan += 1
                continue

            for i in range(len(tokens)-3):

                k = (tp, i)

                pairwise_dist_dic.setdefault(k, []).append(float(tokens[i+3]))

        iterCnt.finish()

    logPrint('[load_pairwise_dist] finished; %d lines contain nan'%cnt_nan)

    #pdb.set_trace()

    return dist_label_dic, pairwise_dist_dic

'''
input: args.pairwise_dist_file
           .dist_cols  e.g. 0,1 draw 0-th and 1st dist metrics
output:
       args.histogram_fig
       - per subplot corresponds to a dist metric.
         histograms wrt pair types (e.g. 0 - same cluster) are drawn.

TBD: types may not be restricted to 0,1.
     types can also be output by load_pairwise_dist
'''
def draw_histogram(args):

    logPrint('[draw_histogram] starts')

    #load data
    dist_label_dic, pairwise_dist_dic = load_pairwise_dist(args.pairwise_dist_file)
    if args.dist_cols == 'na':
        dist_cols = [c for c in dist_label_dic.keys()]
    else:
        dist_cols = [int(col) for col in args.dist_cols.strip().split(',') if col != '']
    #pdb.set_trace()

    #draw histogram
    fig, axes = plt.subplots(nrows=len(dist_cols))
    for i in dist_cols:
        #i_min = min(min(pairwise_dist_dic[(0,i)]), min(pairwise_dist_dic[(1,i)]))
        #i_max = max(max(pairwise_dist_dic[(0,i)]), max(pairwise_dist_dic[(1,i)]))
        axes[i].hist(pairwise_dist_dic[(0, i)], 50, normed=args.normalized, facecolor='green', histtype='step', label='same cluster')
        axes[i].hist(pairwise_dist_dic[(1, i)], 50, normed=args.normalized, facecolor='red', histtype='step', label='different cluster')
        axes[i].set_title('%s'%dist_label_dic[i])
        axes[i].legend()

    #show/save all
    plt.tight_layout()
    plt.savefig(args.histogram_fig) #plt.show()
    #pdb.set_trace()

    logPrint('[draw_histogram] finished. %s written'%args.histogram_fig)

    return

'''
input: args.pairwise_dist_file
           .dist_cols  e.g. 0,1 draw 0-th and 1st dist metrics
           .n_thresholds  # of points per roc curve
output:
       args.roc_fig
       - per curve corresponds to certain dist metric

TBD: types may not be restricted to 0,1.
     types can also be output by load_pairwise_dist
'''
def draw_roc(args):

    logPrint('[draw_roc] starts')

    #load data
    dist_label_dic, pairwise_dist_dic = load_pairwise_dist(args.pairwise_dist_file)
    if args.dist_cols == 'na':
        dist_cols = [c for c in dist_label_dic.keys()]
    else:
        dist_cols = [int(col) for col in args.dist_cols.strip().split(',') if col != '']
    #pdb.set_trace()

    #draw roc
    N = args.n_thresholds
    fig, ax = plt.subplots() #fig, axes = plt.subplots(nrows=len(dist_cols))
    for i in dist_cols:
        i_min = min(min(pairwise_dist_dic[(0,i)]), min(pairwise_dist_dic[(1,i)]))
        i_max = max(max(pairwise_dist_dic[(0,i)]), max(pairwise_dist_dic[(1,i)]))
        T_list = [float(i_max-i_min)/N*t_idx for t_idx in range(N)]
        #pdb.set_trace()

        md_list = [0]*N
        fp_list = [0]*N
        for t_idx in range(N):
            t_val = T_list[t_idx]
            md_list[t_idx] = sum([1 for v in pairwise_dist_dic[(0, i)] if v>t_val]) #seq pair of same cluster has dist greater than threshold (no edge and thus mis-detection)
            fp_list[t_idx] = sum([1 for v in pairwise_dist_dic[(1, i)] if v<t_val]) #seq pair of diff cluster has dist smaller than threshold (w/ edge and thus false positive)
            if args.normalized==1:
                md_list[t_idx] = float(md_list[t_idx]) / len(pairwise_dist_dic[(0, i)])
                fp_list[t_idx] = float(fp_list[t_idx]) / len(pairwise_dist_dic[(1, i)])
        #pdb.set_trace()

        ax.plot(fp_list, md_list, label=dist_label_dic[i], marker='.')
        logPrint('roc for %s done'%dist_label_dic[i])

    ax.set_xlabel('False Positive')
    ax.set_ylabel('Mis-detection')
    ax.legend()

    #show/save all
    plt.tight_layout()
    plt.savefig(args.roc_fig) #plt.show()
    #pdb.set_trace()

    logPrint('[draw_roc] finished. %s written'%args.roc_fig)

    return

'''
Description:
      Based on the learned model, transfer a fasta file into an embedding one

input:
      seq_type     #0: binary  1: dna
      intput_fa    #e.g. sampled_seq_fa file
      model_prefix #learned model
output:
      embed_fa file
'''
def export_embedding(args):

    logPrint('[export_embedding] starts')

    pdt = Predict(args.seq_type, args.model_prefix)

    s2n_obj = seq2nn(pdt.seq_type, pdt.maxlen, pdt.blocklen)
    
    seqs = s2n_obj.transform_seqs_from_fa(args.input_fa); N_seqs = len(seqs)

    #pdb.set_trace()

    with open(args.embed_output, 'w') as fout:

        iterCnt = iterCounter(N_seqs, 'export_embedding')

        for seq in seqs:

            iterCnt.inc()

            seq_embed = pdt.get_embed(seq.tseq)
            seq_embed = seq_embed.flatten()
            #embed_str = np.array2string(seq_embed.flatten(), separator=',')
            embed_str = ','.join(str(v) for v in seq_embed)

            fout.write('>%s\n%s\n'%(seq.description, embed_str))

        iterCnt.finish()

    #pdb.set_trace()

    logPrint('[export_embedding] finished. %s written.'%args.embed_output)

    return

'''
downstream evaluation (histogram and roc) based on:
- pairwise_dist_file
'''
if __name__ == "__main__":

    parser = ArgumentParser()
    subs = parser.add_subparsers()

    if sys.argv[1]=="load_pairwise_dist":
        s_parser = subs.add_parser("load_pairwise_dist")
        s_parser.add_argument('--pairwise_dist_file', type=str, help="pairwise_dist_file")
        args = parser.parse_args()
        dist_label_dic, pairwise_dist_dic = load_pairwise_dist(args.pairwise_dist_file)
        pdb.set_trace()

    elif sys.argv[1]=="draw_histogram":
        s_parser = subs.add_parser("draw_histogram")
        s_parser.add_argument('--pairwise_dist_file', type=str, help="input of pairwise dist file")
        s_parser.add_argument('--histogram_fig', type=str, help="output of histogram fig file")
        s_parser.add_argument('--dist_cols', type=str, default='na', help="list of dist col indice e.g. 0,1 etc; default is na - sel all available cols")
        s_parser.add_argument('--normalized', type=int, default=0, help="normalize histogram (1) or not (0 - default)")
        args = parser.parse_args()
        draw_histogram(args)
        #pdb.set_trace()

    elif sys.argv[1]=="draw_roc":
        s_parser = subs.add_parser("draw_roc")
        s_parser.add_argument('--pairwise_dist_file', type=str, help="input of pairwise dist file")
        s_parser.add_argument('--roc_fig', type=str, help="output of roc fig file")
        s_parser.add_argument('--dist_cols', type=str, default='na', help="list of dist col indice e.g. 0,1 etc; default is na - sel all available cols")
        s_parser.add_argument('--normalized', type=int, default=0, help="normalize roc (1) or not (0 - default)")
        s_parser.add_argument('--n_thresholds', type=int, default=100, help="number of thresholds/points for roc curve")
        args = parser.parse_args()
        draw_roc(args)
        #pdb.set_trace()

    elif sys.argv[1]=="export_embedding":
        s_parser = subs.add_parser("export_embedding")
        s_parser.add_argument("--seq_type", type=int, help="type of seq. 0: binary seq; 1: dna/atcg seq")
        s_parser.add_argument("--input_fa", type=str, help="input seqs in fasta format")
        s_parser.add_argument("--embed_output", type=str, help="Output file of embed_fa_format")
        s_parser.add_argument("--model_prefix", type=str, help="learned model prefix")
        args = parser.parse_args()

        export_embedding(args)

        #pdb.set_trace()


    else:
        pdb.set_trace()
