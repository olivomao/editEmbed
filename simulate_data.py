'''
comprehensive data simulation
'''

from argparse import ArgumentParser
import pdb
import sys
import random
import numpy as np
import os
from Bio import SeqIO

from util import *
from edit_dist import *
from inference import Predict, seq2nn

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

    #weights are normalized
    if args.weight_distr == 1:
        weights = np.random.uniform(0, 1, args.num)
        weights = weights / sum(weights)

    with open(args.output, 'w') as of:

        for i in range(args.num):

            #pdb.set_trace()

            sid = "%s%d"%(args.sid_pre, i)
            length = args.length

            if args.weight_distr == 0:
                weight = 1.0 / args.num
            else:
                weight = weights[i] #np.random.uniform(0,1)

            seq_description = ">%s\tgene=%s\tweight=%f"%(sid, sid, weight)
            
            if args.seq_type == 0:
                seq = random_bin_string(length)
            else:
                seq = random_dna_string(length)


            of.write("%s\n%s\n"%(seq_description, seq))

    logPrint("[gen_cluster_center] %s written"%args.output)

    return

def sample_from_cluster(args):

    logPrint("[sample_from_cluster]")

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

    fa_input = args.fa_input
    fa_output = args.fa_output
    prefix = args.prefix
    seq_type = args.type
    total_samples = args.total_samples

    #pdb.set_trace()
    if args.thread == 1:
        #pdb.set_trace()
        cmd = 'python indel_channel.py %s %s --total_samples %d --copy %d --ins %f --dele %f --sub %f --output %s --type %d'% \
              (fa_input, prefix, total_samples, args.copy, args.ins, args.dele, args.sub, fa_output, seq_type)
        run_cmd(cmd)
        logPrint('%s written'%fa_output)
    else:
        #pdb.set_trace()
        path, fn = os.path.split(fa_output)
        tmp_dir = path + '/tmp_split_clusters/'
        tmp_dir2 = path + '/tmp_split_noisy_samples/'
        cmd = 'mkdir -p %s'%tmp_dir; run_cmd(cmd)
        cmd = 'mkdir -p %s'%tmp_dir2; run_cmd(cmd)

        tot_clusters = os.popen('grep \'>\' %s | wc -l'%(fa_input)).read()
        tot_clusters = int(tot_clusters.split()[0])
        num_cluster_per_split = int(tot_clusters/args.thread)+1;
        split_fa_file(fa_input, out_dir=tmp_dir, num_seq_per_file=num_cluster_per_split)

        split_inputs = os.listdir(tmp_dir); split_inputs.sort()
        cmds = []
        split_prefix_idx = 0
        for split_input in split_inputs:
            split_input_path = tmp_dir + '/' + split_input
            split_output_path = tmp_dir2 + '/' + split_input
            sub_prefix = prefix + '_p%d'%split_prefix_idx
            split_prefix_idx += 1

            cmd = 'python indel_channel.py %s %s --total_samples %d --copy %d --ins %f --dele %f --sub %f --output %s  --type %d'% \
                  (split_input_path, sub_prefix, total_samples, args.copy, args.ins, args.dele, args.sub, split_output_path, seq_type)
            cmds.append(cmd)

        #pdb.set_trace()
        run_cmds(cmds, args.thread)

        #pdb.set_trace()

        merge_files(tmp_dir2, fa_output)
        #pdb.set_trace()

        logPrint('%s written'%fa_output)

        #remove merge_files
        if args.clear_split_files == 1:
            run_cmd("rm -r %s"%tmp_dir)
            run_cmd("rm -r %s"%tmp_dir2)
            logPrint("intermediate split files cleared")

    logPrint("[sample_from_cluster] finished")

    return


'''
describe two seq relationship

input:
tid_gid_1: [trid, gene_id] of seq1
tid_gid_2: [trid, gene_id] of seq2

type:   0: sampled_seqs have same tid (e.g. cluster)
        1: different tid (different gene)
        2: different tid (same gene) ==> to be addressed when dealing with real data
'''
def check_type(tid_gid_1, tid_gid_2):

    if tid_gid_1[0]==tid_gid_2[0]:
        tp = 0 #different noisy sample, same cluster center (isoform)
    elif tid_gid_1[1]!=tid_gid_2[1]:
        tp = 1 #different genes
    else:
        tp = 2 #different isoform, same gene

    return tp

Base2Index_DNA = {'A':0, 'C':1, 'T':2, 'G':3}
Base2Index_Bin = {'0':0, '1':1}
def cgk_embedding(s, N, R, seq_type=0):

    #pdb.set_trace()

    if seq_type==1:
        s = s.upper()

    #L_R = 3*N*4
    #R = np.random.randint(0,2,L_R)
    
    i = 0
    s1 = ""
    for j in range(3*N):
        if i<len(s):
            c = s[i]
            s1 = s1 + c
            if seq_type==0:
                i = i + R[(j-1)*2+Base2Index_Bin[c]]
            else:
                i = i + R[(j-1)*4+Base2Index_DNA[c]]
        else:
            s1 = s1 + "N"
    return s1

def hamming(a,b):
    return sum([1 for i in range(len(a)) if a[i]!=b[i]])

'''
proj a --> a1, and b --> b1
to use ham(a1, b1) to bound edit(a,b)
'''
def proj_hamming_dist(a, b, seq_type):
    #pdb.set_trace()
    N_iter = 10
    N = max(len(a), len(b))
    min_dist = 3*N+1
    for i in range(N_iter):

        if seq_type==0:
            L_R = 3*N*2
        else:
            L_R = 3*N*4
        R = np.random.randint(0,2,L_R)

        a1 = cgk_embedding(a, N, R, seq_type)
        b1 = cgk_embedding(b, N, R, seq_type)
        h_a1_b1 = hamming(a1, b1)#a1,b1 of 3N lengths
        #print(h_a1_b1)
        if h_a1_b1<min_dist:
            min_dist = h_a1_b1

    min_dist = float(min_dist)/(3*N)
    #print(min_dist)
    return min_dist

'''
calc dist w/o loading learned model

dist is decided by distance_type
- 0: edit dist
- 1: gapped_edit dist
- 2: proj_hamming dist
- 3: nn_distance (deep learning based)

seq_type: 0 binary 1 dna
'''
def calc_non_learned_dist(distance_type,
                          seq1_str,
                          seq2_str,
                          seq_type):
    if distance_type==0:
        ed = edit_dist(seq1_str, seq2_str) #seq type independent
    elif distance_type==1:
        pdb.set_trace()
        ed = gapped_edit_dist(seq1_str, seq2_str) #seq type independent
    elif distance_type==2:
        ed = proj_hamming_dist(seq1_str, seq2_str, seq_type) #seq type dependent
    return ed

'''
single-thread

calc pairwise dist b/w seqs in fa_1 and seqs in fa_2
output res in dist_out (pairwise distance file format)

dist is decided by distance_type
- 0: edit dist
- 1: gapped_edit dist
- 2: proj_hamming dist
- 3: nn_distance (deep learning based)

seq_type:
- 0: binary seq
- 1: dna (ATCG) seq

==== Note:

for seq pairs, we do not consider:
- seq1 and seq2 of same name
- seq1 or seq2 is empty

model_prefix: to be used when distance_type==3
              incorporated into: nn_dist_args.model_prefix         #model prefix ...ckpt to be loaded

max_num_dist_1thread: set max seq pairs for output. mainly used for debugging purpose
'''
distance_type_names = ['edit_dist', 'gapped_edit_dist', 'proj_hamming_dist', 'nn_distance']


def calc_dist_1thread(distance_type_list, seq_type, fa_1, fa_2, dist_out, addheader=0,
                      model_prefix='NA', max_num_dist_1thread=-1):

    #pdb.set_trace()
    if 3 in distance_type_list: 
        pdt = Predict(seq_type, model_prefix)
        #print('pdt seq type '+ str(seq_type))
        s2n_obj = seq2nn(pdt.seq_type, pdt.maxlen, pdt.blocklen) #s2n_obj = seq2nn(seq_type, 500, 10)
        seqs1 = s2n_obj.transform_seqs_from_fa(fa_1); N_seqs1 = len(seqs1)
        seqs2 = s2n_obj.transform_seqs_from_fa(fa_2); N_seqs2 = len(seqs2)
        #pdt = Predict(seq_type, model_prefix)
    else: 
        seqs1 = list(SeqIO.parse(fa_1, 'fasta')); print('%d seqs loaded'%len(seqs1)); N_seqs1 = len(seqs1)
        seqs2 = list(SeqIO.parse(fa_2, 'fasta')); print('%d seqs loaded'%len(seqs2)); N_seqs2 = len(seqs2)

    if fa_1==fa_2:
        N_tot = (N_seqs1-1)*N_seqs1/2;
    else:
        N_tot = N_seqs2 * N_seqs1;

    fo = open(dist_out, 'w')

    if addheader==1:
        #pdb.set_trace()
        headerline = '#seq_id1\tseq_id2\ttype\t' #%distance_type_names[distance_type]
        for distance_type in distance_type_list:
            headerline += '%s\t'%distance_type_names[distance_type]
        fo.write('%s\n'%headerline)

    iterCnt = iterCounter(N_tot, "calc_dist_1thread") #: %s and %s\n"%(fa_1, fa_2))

    cnt = 0

    for seq1_idx in range(N_seqs1):
        
        if fa_1==fa_2:
            seq2_idx_range = range(seq1_idx, N_seqs2)
        else:
            seq2_idx_range = range(N_seqs2)

        for seq2_idx in seq2_idx_range: 
            seq1 = seqs1[seq1_idx]
            seq2 = seqs2[seq2_idx]

            iterCnt.inc()          

            #check type
            #pdb.set_trace()

            tokens1 = seq1.description.split() 
            tokens2 = seq2.description.split()

            seq1_id = tokens1[0]
            seq1_trid = tokens1[1][4:] #skip tid=
            seq1_gene_id = tokens1[2][5:] #skip gene=

            seq2_id = tokens2[0]
            seq2_trid = tokens2[1][4:]
            seq2_gene_id = tokens2[2][5:]

            if seq1_id == seq2_id: continue

            #calc dist
            seq1_str = str(seq1.seq)
            seq2_str = str(seq2.seq)
            if seq1_str=='' or seq2_str=='': continue

            cnt+=1
            if max_num_dist_1thread != -1 and cnt>max_num_dist_1thread:
                break

            distance_list = []
            #pdb.set_trace()
            for distance_type in distance_type_list:
                if distance_type!=3:
                    ed = calc_non_learned_dist(distance_type, seq1_str, seq2_str, seq_type)
                elif distance_type==3:
                    #pdb.set_trace()
                    tseq1 = seqs1[seq1_idx].tseq
                    tseq2 = seqs2[seq2_idx].tseq
                    ed = pdt.predict0(tseq1, tseq2)
                    ed = np.mean(ed)
                distance_list.append(ed)

            #check pair of (x1,x2) type
            tp = check_type([seq1_trid, seq1_gene_id], [seq2_trid, seq2_gene_id])

            line_content = '%s\t%s\t%s\t'%(seq1_id, seq2_id, tp)
            #pdb.set_trace()
            try:
                for ed in distance_list:
                    line_content += '%f\t'%ed 
                fo.write('%s\n'%line_content) #pairwise_distance format
            except:
                pdb.set_trace()
        #for seq2_idx
        if max_num_dist_1thread != -1 and cnt>max_num_dist_1thread:
                break
    #for seq1_idx

    iterCnt.finish()

    logPrint("calc_dist_1thread: %s and %s finished. %s written\n"%(fa_1, fa_2, dist_out));

    fo.close()

    return


'''
multi-thread

- model_prefix: only used when distance_type_list contains nn_dist
'''
def calc_dist_Nthread(distance_type_list_str, seq_type, seq_fa, dist_out, thread, addheader, clear_intermediate, model_prefix='NA', max_num_dist_1thread=-1 ):

    sub_fld = {0:'edit', 1:'gapped_edit', 2:'proj_hamming', 3:'nn_distance'}

    fld, fn = os.path.split(dist_out)
    tmp_fld_split = '%s/split_samples/'%(fld); run_cmd('mkdir -p %s'%tmp_fld_split)
    tmp_fld_dist  = '%s/split_dist/'%(fld); run_cmd('mkdir -p %s'%tmp_fld_dist)

    num_seq = int(os.popen('grep \'>\' %s | wc -l'%(seq_fa)).read().split()[0])
    num_seq_per_split = max(int(num_seq/np.sqrt(2*thread)), 1)

    #pdb.set_trace()

    split_fa_file(seq_fa, out_dir=tmp_fld_split, num_seq_per_file=num_seq_per_split)

    split_fs = os.listdir(tmp_fld_split)
    split_fs.sort()
    split_fs = [tmp_fld_split+'/'+i for i in split_fs]
    N_split_fs = len(split_fs)

    #pdb.set_trace()

    '''
    for i in range(N_split_fs):
        for j in xrange(i, N_split_fs):
            split_f1 = split_fs[i]
            split_f2 = split_fs[j]
            dist_f1_f2 = tmp_fld_dist + '/' + 'i_%05d_j_%05d.dist'%(i,j)
            main_dist_edit_1thread(split_f1, split_f2, dist_f1_f2)
    '''
    cmds = []
    for i in range(N_split_fs):
        for j in xrange(i, N_split_fs):
            split_f1 = split_fs[i]
            split_f2 = split_fs[j]
            dist_f1_f2 = tmp_fld_dist + '/' + 'i_%05d_j_%05d.dist'%(i,j)
            #pdb.set_trace()
            cmd = 'python simulate_data.py calc_dist_1thread '+\
                  '--distance_type_list %s --seq_type %d --seq_fa1 %s --seq_fa2 %s --dist_out %s --addheader %d --model_prefix %s --max_num_dist_1thread %d'%\
                  (distance_type_list_str, seq_type, split_f1, split_f2, dist_f1_f2, 0, model_prefix, max_num_dist_1thread)
            cmds.append(cmd)
    #pdb.set_trace()
    run_cmds(cmds, thread)

    #if addheader==0:
    #    headerline = ''
    #else:
    #    headerline = '#seq_id1\tseq_id2\ttype\t%s\n'%distance_type_names[distance_type]
    merge_files(tmp_fld_dist, dist_out, dele=False, headerline='')

    cmd = 'sort -k1,1 -k2,2 %s > %s.tmp'%(dist_out, dist_out)
    run_cmd(cmd)

    cmd = 'rm %s'%dist_out
    run_cmd(cmd)

    if addheader==0:
        #pdb.set_trace()
        cmd = 'mv %s.tmp %s'%(dist_out, dist_out)
        run_cmd(cmd)
    else:
        #pdb.set_trace()

        with open('%s.tmp2'%dist_out, 'w') as fo_header:
            #pdb.set_trace()
            headerline = '#seq_id1\tseq_id2\ttype\t' #%distance_type_names[distance_type]
            for distance_type in distance_type_list_str.split(','):
                distance_type = int(distance_type)
                headerline += '%s\t'%distance_type_names[distance_type]
            fo_header.write('%s\n'%headerline)

        cmd = 'cat %s.tmp2 %s.tmp > %s'%(dist_out, dist_out, dist_out)
        run_cmd(cmd)

        cmd = 'rm %s.tmp %s.tmp2'%(dist_out, dist_out)
        run_cmd(cmd)
    
    if clear_intermediate==1:
        run_cmd('rm -r %s'%tmp_fld_split)
        run_cmd('rm -r %s'%tmp_fld_dist)

    logPrint('[calc_dist_Nthread] %s written'%dist_out)
    
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

    python simulate_data.py sample_from_cluster --fa_input cluster_center_fa
                                             [--type tp]                    # type 0: binary and 1: dna/ATCG
                                             --fa_output sampled_seq_fa
                                             --prefix prefix                # actual prefix may also contain thread idx
                                             [--total_samples -1]           # (default; use --copy) or >0 (use cluster weight)
                                             [--copy >0]                    # (default 1 used if --total_samples -1)
                                             [--ins insertion_rate]         # default 0
                                             [--dele delete_rate]           # default 0
                                             [--sub substitution_rate]      # default 0
                                             [--thread num_processes]       # default 1
                                             [--clear_split_files 0/1]      # default 0
    

(3) calculate distances

    (3.1) 1-thread case:

    python simulate_data.py calc_dist_1thread  --distance_type_list  dt0[,dt1,..]           #distance_type: 0.edit 1.gapped_edit 2.proj_hamming 3.nn_distance (deep learning based)
                                               --seq_type      st           #type of seq. 0: binary seq; 1: dna/atcg seq
                                               --seq_fa1       f1          
                                               --seq_fa2       f2
                                               --dist_out      do           #Output file of pairwise distance (e.g. pairwise_distance_format)
                                               [--addheader    ah]          #Add headerline (1) or not (0, default) to the output
                                               [--model_prefix mp/NA]          #needed if nn_distance to be estimated; otherwise NA (default)
                                               [--max_num_dist_1thread -1]  #max number of pairwise dist to calculate; default -1; used for test purpose


    (3.2) N-thread case:

    python simulate_data.py calc_dist --distance_type_list       dt0[,dt1,..]              #distance_type: 0.edit 1.gapped_edit 2.proj_hamming 3.nn_distance (deep learning based)
                                      --seq_type            st              #type of seq. 0: binary seq; 1: dna/atcg seq
                                      --seq_fa              f               #Input seqs in sampled_seq_fa format
                                      --dist_out            do
                                      --thread              num_thread
                                      [--addheader          ah/0]
                                      [--clear_intermediate ci/0]
                                      [--model_prefix       mp/NA]
                                      [--max_num_dist_1thread -1]  #max number of pairwise dist to calculat
e; default -1; used for test purpose

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

    elif sys.argv[1]=='sample_from_cluster':

        s_parser = subs.add_parser('sample_from_cluster')

        s_parser.add_argument("--fa_input")
        s_parser.add_argument("--type", type=int, default=0, help="type of seq. 0 is binary; 1 is dna e.g. ATCG")
        s_parser.add_argument("--fa_output")
        s_parser.add_argument("--prefix", help="prefix for sampled seq. Actual prefix will also contain thread info if thread > 1")
        s_parser.add_argument("--total_samples", type=int, default=-1, help="Total number of sampled sequences. Default: -1 "+\
                              "sample per cluster by --copy; >0, sample per cluster by its relative weight")
        s_parser.add_argument("--copy", type=int, default=1, help="Only used if --total_samples -1 "+\
                              "; Number of copies to simulate per cluster (default: 1)")
        s_parser.add_argument("--ins", "-i", type=float, default=0, help="Insert error rate [0-1] (default: 0)")
        s_parser.add_argument("--dele", "-d", type=float, default=0, help="Deletion error rate [0-1] (default: 0)")
        s_parser.add_argument("--sub", "-s", type=float, default=0, help="Substitution error rate [0-1] (default: 0)")
        s_parser.add_argument("--thread", "-p", type=int, default=1, help="specify number of processes to use")
        s_parser.add_argument("--clear_split_files", type=int, default=0, help="default 0; 1: clear intermediate split files")

        args = parser.parse_args()

        sample_from_cluster(args)

    elif sys.argv[1]=='calc_dist':

        s_parser = subs.add_parser('calc_dist')

        #s_parser.add_argument("--distance_type", type=int, default=0, help="distance_type: 0.edit 1.gapped_edit 2.proj_hamming 3.nn_distance (deep learning based)")
        s_parser.add_argument("--distance_type_list", type=str, help="list of distance_types. e.g. 0,3. distance_type: 0.edit 1.gapped_edit 2.proj_hamming 3.nn_distance (deep learning based)")
        s_parser.add_argument("--seq_type", type=int, help="type of seq. 0: binary seq; 1: dna/atcg seq")
        s_parser.add_argument("--seq_fa", help="Input seqs in sampled_seq_fa format")
        s_parser.add_argument("--dist_out", help="Output file of pairwise distance (e.g. pairwise_distance_format)")
        s_parser.add_argument("--thread", type=int, default=1, help="number of processes")
        s_parser.add_argument("--addheader", type=int, default=0, help="Add headerline (1) or not (0, default) to the output file of pairwise distance (e.g. pairwise_distance_format)")
        s_parser.add_argument("--clear_intermediate", type=int, default=0, help="clear intermediate files (1) or not (0)")
        s_parser.add_argument("--model_prefix", default="NA", type=str, help="model prefix to be used if nn_distance needs to be estimated. Default is NA")
        s_parser.add_argument("--max_num_dist_1thread", default=-1, type=int, help="max number of pairwise dist to calculate; default -1; used for test purpose")

        args = parser.parse_args()

        if args.thread==1:
            #pdb.set_trace()
            distance_type_list = [int(dt) for dt in args.distance_type_list.split(',') if dt!='']
            calc_dist_1thread(distance_type_list, 
                              args.seq_type,
                              args.seq_fa,
                              args.seq_fa,
                              args.dist_out,
                              addheader=args.addheader,
                              model_prefix=args.model_prefix,
                              max_num_dist_1thread=args.max_num_dist_1thread)
        else:
            #pdb.set_trace()
            calc_dist_Nthread(args.distance_type_list,
                              args.seq_type,
                              args.seq_fa,
                              args.dist_out,
                              args.thread,
                              args.addheader,
                              args.clear_intermediate,
                              args.model_prefix,
                              args.max_num_dist_1thread)

    elif sys.argv[1]=='calc_dist_1thread': #to be used by calc_dist_Nthread

        s_parser = subs.add_parser('calc_dist_1thread')

        #s_parser.add_argument("--distance_type", type=int, help="distance_type: 0.edit 1.gapped_edit 2.proj_hamming 3.nn_distance (deep learning based)")
        s_parser.add_argument("--distance_type_list", type=str, help="list of distance_types. e.g. 0,3. distance_type: 0.edit 1.gapped_edit 2.proj_hamming 3.nn_distance (deep learning based)")
        s_parser.add_argument("--seq_type", type=int, help="type of seq. 0: binary seq; 1: dna/atcg seq")
        s_parser.add_argument("--seq_fa1", type=str, help="First input seqs in sampled_seq_fa format")
        s_parser.add_argument("--seq_fa2", type=str, help="Second input seqs in sampled_seq_fa format. only consider non-dup (s1,s2)")
        s_parser.add_argument("--dist_out", type=str, help="Output file of pairwise distance (e.g. pairwise_distance_format)")
        s_parser.add_argument("--addheader", type=int, default=0, help="Add headerline (1) or not (0, default) to the output file of pairwise distance (e.g. pairwise_distance_format)")
        s_parser.add_argument("--model_prefix", default="NA", type=str, help="model prefix to be used if nn_distance needs to be estimated. Default is NA")
        s_parser.add_argument("--max_num_dist_1thread", default=-1, type=int, help="max number of pairwise dist to calculate; default -1; used for test purpose")
        args = parser.parse_args()

        #pdb.set_trace()
        distance_type_list = [int(dt) for dt in args.distance_type_list.split(',') if dt!='']
        calc_dist_1thread(distance_type_list, 
                          args.seq_type,
                          args.seq_fa1,
                          args.seq_fa2,
                          args.dist_out,
                          addheader=args.addheader,
                          model_prefix=args.model_prefix,
                          max_num_dist_1thread=args.max_num_dist_1thread)
    
    else:

        pdb.set_trace()
