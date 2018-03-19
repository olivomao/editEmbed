from argparse import ArgumentParser
import pdb, sys
import numpy as np

import matplotlib #draw_hist issue
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from util import logPrint, iterCounter


'''
input:
- pairwise_dist_file

output:
- pairwise_dist_dic
  key: (pair_type, dist_col_idx) val: list of dist vals
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

    with open(pairwise_dist_file, 'r') as fin:

        for line in fin:

            iterCnt.inc()

            if line[0]=='#': continue

            tokens = line.strip().split()

            tp = int(tokens[2])

            for i in range(len(tokens)-3):

                k = (tp, i)

                pairwise_dist_dic.setdefault(k, []).append(float(tokens[i+3]))

        iterCnt.finish()

    logPrint('[load_pairwise_dist] finished')

    #pdb.set_trace()

    return dist_label_dic, pairwise_dist_dic

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

    else:
        pdb.set_trace()
