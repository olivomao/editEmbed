from argparse import ArgumentParser
import pdb, sys, copy, multiprocessing, os

from util import logPrint, run_cmd
from train import train_seq2seq, train_siamese_seq2seq

'''
This .py file tries to make the train/eval more flexible and efficient
- include train and eval procedure
- it takes a config_param file, do related train or eval, stores res in pre-defined location structure
'''

'''
functions
'''

'''
input: config_file
       e.g.
       A      a1;a2
       B      b1

output: list of [desc, dic w/ key of label and val of val]
       e.g.
       [ [A_a1_B_b1, dic{A: a1, B: b1}],
         [A_a2_B_b1, dic{A: a2, B: b1}]
       ]
Note:
- a1,a2 is not param combination, need a1;a2
- lines starting w/ '#' in config_file are skipped
- if desc=='single_set', there is no param combinations
'''
def parse_config_file(config_file):

    desc_dic_list = []

    itms = [] #list of [key, list of vals]
    shared_dic = {}
    with open(config_file, 'r') as cf:
        for line in cf:
            if line[0]=='#': continue
            tokens = line.strip().split()
            if len(tokens)<2: continue
            key = tokens[0]
            vals = [v for v in tokens[1].split(';') if v!='']
            if(len(vals)==1):
                shared_dic[key]=vals[0]
            else:
                itms.append([key, vals])

    #pdb.set_trace()

    if len(itms)==0:
        return [['single_set', shared_dic]]

    else:
        def update(shared_dic, desc, itms, idx, desc_dic_list):
            if idx==len(itms):
                desc_dic_list.append([desc[:-1], copy.deepcopy(shared_dic)])
                return
            else:
                for v in itms[idx][1]:
                    desc_copy = desc + '%s_%s_'%(itms[idx][0], v)
                    shared_dic[itms[idx][0]]=v
                    update(shared_dic, desc_copy, itms, idx+1, desc_dic_list)
        update(shared_dic, '', itms, 0, desc_dic_list)
        #for desc_dic in desc_dic_list:
        #    print desc_dic
        #pdb.set_trace()
        return desc_dic_list

'''
batch train one set of parameters

param_dic: key: param_label and val: string of param val
param_desc: some params in param_dic are one case of possible combinations, param_desc describes this
            for example, param_desc=hdsz_100_learning_rate_0.001
args: initial args passed to batch_test.py for control purpose 
'''
def batch_test_train_1job(input_args):

    param_dic, param_desc, args = input_args #packed to input_args for parallel purpose

    #pdb.set_trace()

    logLabel = '[train_1job_%s] '%param_desc

    logPrint(logLabel+'Start')

    root_dir = param_dic['root_dir']
    data_label = param_dic['data_label']
    model_label = param_dic['model_label']

    data_dir = '%s/%s/train/data/'%(root_dir, data_label)
    run_cmd('mkdir -p %s'%data_dir); logPrint('%s created'%data_dir)
    model_dir = '%s/%s/train/%s/%s/'%(root_dir, data_label, model_label, param_desc)
    run_cmd('mkdir -p %s'%model_dir); logPrint('%s created'%model_dir)

    dst_config = '%s/%s/train/%s/config.txt'%(root_dir, data_label, model_label)
    if os.path.exists(dst_config)==False:
        run_cmd('cp %s %s'%(args.config_file, dst_config))

    sample_fa = '%s/sample.fa'%data_dir
    sample_dist = '%s/sample.dist'%data_dir

    cmd = 'python train.py use_simulate_data '+\
                          '--input_type                 1 '+\
                          '--train_input1               %s '%sample_fa+\
                          '--train_input2               %s '%sample_dist+\
                          '--seq_type                   %s '%param_dic['seq_type']+\
                          '--train_output_dir           %s '%model_dir+\
                          '--max_num_to_sample          %s '%param_dic['max_num_to_sample']+\
                          '--batch_size                 %s '%param_dic['batch_size']+\
                          '--num_epochs                 %s '%param_dic['num_epochs']+\
                          '--load_model                 %s '%param_dic['load_model']+\
                          '--maxlen                     %s '%param_dic['maxlen']+\
                          '--blocklen                   %s '%param_dic['blocklen']+\
                          '--embedding_size             %s '%param_dic['embedding_size']+\
                          '--num_layers                 %s '%param_dic['num_layers']+\
                          '--hidden_sz                  %s '%param_dic['hidden_sz']+\
                          '--learning_rate              %s '%param_dic['learning_rate']+\
                          '--dropout                    %s '%param_dic['dropout']
    #pdb.set_trace()
    run_cmd(cmd)

    logPrint(logLabel+'End')

    return

'''
batch train one set of parameters (for seq2seq architecture)

param_dic: key: param_label and val: string of param val
param_desc: some params in param_dic are one case of possible combinations, param_desc describes this
            for example, param_desc=hdsz_100_learning_rate_0.001
args: initial args passed to batch_test.py for control purpose 
'''
def batch_test_train_1job_seq2seq(input_args):

    #pdb.set_trace()

    param_dic, param_desc, args = input_args #packed to input_args for parallel purpose 
    logLabel = '[train_1job_seq2seq_%s] '%param_desc

    logPrint(logLabel+'Start')

    root_dir = param_dic['root_dir']
    data_label = param_dic['data_label']
    model_label = param_dic['model_label']

    data_dir = '%s/%s/train/data/'%(root_dir, data_label)
    run_cmd('mkdir -p %s'%data_dir); logPrint('%s created'%data_dir)
    model_dir = '%s/%s/train/%s/%s/'%(root_dir, data_label, model_label, param_desc)
    run_cmd('mkdir -p %s'%model_dir); logPrint('%s created'%model_dir)

    dst_config = '%s/%s/train/%s/config.txt'%(root_dir, data_label, model_label)
    if os.path.exists(dst_config)==False:
        run_cmd('cp %s %s'%(args.config_file, dst_config))

    #sample_fa = '%s/sample.fa'%data_dir
    #sample_dist = '%s/sample.dist'%data_dir
    seq2seq_pair = '%s/seq2seq_cgk.txt'%data_dir

    #to be passed to train_seq2seq function
    param_dic['seq2seq_pair_path'] = seq2seq_pair
    param_dic['model_dir_path'] = model_dir

    if param_dic['n_clusters_validation']>0:
        #pdb.set_trace()
        param_dic['seq2seq_pair_path_validation'] = '%s/seq2seq_cgk_validation.txt'%data_dir

    train_seq2seq(param_dic) 

    logPrint(logLabel+'End')

    return


'''
modified based on batch_test_train_1job_seq2seq

batch train one set of parameters (for siamese seq2seq architecture)

param_dic: key: param_label and val: string of param val
param_desc: some params in param_dic are one case of possible combinations, param_desc describes this
            for example, param_desc=hdsz_100_learning_rate_0.001
args: initial args passed to batch_test.py for control purpose 
'''
def batch_test_train_1job_siamese_seq2seq(input_args):

    #pdb.set_trace()

    param_dic, param_desc, args = input_args #packed to input_args for parallel purpose 
    logLabel = '[train_1job_siamese_seq2seq_%s] '%param_desc

    logPrint(logLabel+'Start')

    root_dir = param_dic['root_dir']
    data_label = param_dic['data_label']
    model_label = param_dic['model_label']

    data_dir = '%s/%s/train/data/'%(root_dir, data_label)
    run_cmd('mkdir -p %s'%data_dir); logPrint('%s created'%data_dir)
    model_dir = '%s/%s/train/%s/%s/'%(root_dir, data_label, model_label, param_desc)
    run_cmd('mkdir -p %s'%model_dir); logPrint('%s created'%model_dir)

    dst_config = '%s/%s/train/%s/config.txt'%(root_dir, data_label, model_label)
    if os.path.exists(dst_config)==False:
        run_cmd('cp %s %s'%(args.config_file, dst_config))

    #sample_fa = '%s/sample.fa'%data_dir
    #sample_dist = '%s/sample.dist'%data_dir
    #seq2seq_pair = '%s/seq2seq_cgk.txt'%data_dir
    siamese_seq2seq = '%s/siamese_seq2seq.txt'%data_dir

    #to be passed to train_seq2seq function
    #param_dic['seq2seq_pair_path'] = seq2seq_pair
    param_dic['siamese_seq2seq'] = siamese_seq2seq
    param_dic['model_dir_path'] = model_dir

    if param_dic['n_clusters_validation']>0:
        #pdb.set_trace()
        param_dic['siamese_seq2seq_validation'] = '%s/siamese_seq2seq_validation.txt'%data_dir

    train_siamese_seq2seq(param_dic) 

    logPrint(logLabel+'End')

    return

'''
Note:
Parse a config_file (e.g. hdsz=100,200) into multiple training jobs (e.g. job of hdsz=100, job of hdsz=200)
Parallelly train these jobs, and save model into locations specified in storage_structure

to train 1 job (1 seq of hyparams), use:
    batch_test_train_1job for siamese architecture
    batch_test_train_1job_seq2seq for seq2seq architecture
'''
def batch_test_train(args):

    logPrint('[batch_test_train] Start')

    config_desc_dic_list = parse_config_file(args.config_file)
    #pdb.set_trace()

    logPrint('[batch_test_train] %s parsed'%args.config_file)

    nJobs = args.N 

    if nJobs==1:
        for param_desc, param_dic in config_desc_dic_list:
            if args.a==0: #siamese
                batch_test_train_1job((param_dic, param_desc, args))
            elif args.a==1: #seq2seq
                #pdb.set_trace()
                batch_test_train_1job_seq2seq((param_dic, param_desc, args))
            elif args.a==2: #siamese seq2seq
                batch_test_train_1job_siamese_seq2seq((param_dic, param_desc, args))
    else:
        args_list = [(param_dic, param_desc, args) for param_desc, param_dic in config_desc_dic_list]
        p = multiprocessing.Pool(nJobs)
        if args.a==0: #siamese
            p.map(batch_test_train_1job, args_list)
        elif args.a==1: #seq2seq
            p.map(batch_test_train_1job_seq2seq, args_list)
        elif args.a==2: #siamese seq2seq
            p.map(batch_test_train_1job_siamese_seq2seq, args_list)

    logPrint('[batch_test_train] End')

    return


'''
batch evaluate one set of parameters

param_dic: key: param_label and val: string of param val
param_desc: some params in param_dic are one case of possible combinations, param_desc describes this
            for example, param_desc=hdsz_100_learning_rate_0.001
args: initial args passed to batch_test.py for control purpose

Note:
- it's possible that there's no trained model (due to training issue such as loss=nan) and in this case just return

'''
def batch_test_eval_1job(input_args):

    #pdb.set_trace()

    param_dic, param_desc, args = input_args #packed to input_args for parallel purpose

    #pdb.set_trace()

    logLabel = '[eval_1job_%s] '%param_desc

    logPrint(logLabel+'Start')

    #---------- locations
    root_dir = param_dic['root_dir']
    data_label = param_dic['data_label']
    model_label = param_dic['model_label']

    data_dir = '%s/%s/eval/data/'%(root_dir, data_label)
    run_cmd('mkdir -p %s'%data_dir); logPrint('%s created'%data_dir)
    model_train_dir = '%s/%s/train/%s/%s/'%(root_dir, data_label, model_label, param_desc)
    model_eval_dir = '%s/%s/eval/%s/%s/'%(root_dir, data_label, model_label, param_desc)
    run_cmd('mkdir -p %s'%model_eval_dir); logPrint('%s created'%model_eval_dir)

    sample_fa = '%s/sample.fa'%data_dir

    #---------- select a trained model (i.e. a check point)
    ckpt_path, ckpt_name, step_loss_fn_list = select_ckpt(model_train_dir)

    if ckpt_path == '':
        logPrint(logLabel+'End (no valid ckpt found)')
        #pdb.set_trace()
        return
    else:
        ckpt_dir = '%s/%s/'%(model_eval_dir, ckpt_name)
        run_cmd('mkdir -p %s'%ckpt_dir); logPrint('%s created'%ckpt_dir)
        sample_dist = '%s/sample.dist'%ckpt_dir
        #pdb.set_trace()
    
    #---------- proc eval job

    tasks = [int(t) for t in args.tasks.split(',') if t != '']

    # dist
    cmd = 'python simulate_data.py calc_dist '+\
                        '--distance_type_list %s '%param_dic['dist_tp_list_eval'] +\
                        '--seq_type %s '%param_dic['seq_type'] +\
                        '--seq_fa %s '%sample_fa +\
                        '--dist_out %s '%sample_dist+\
                        '--thread %d '%args.N +\
                        '--addheader %s '%param_dic['add_hd_eval'] +\
                        '--clear_intermediate %s '%param_dic['clear_interm_eval'] +\
                        '--model_prefix %s '%ckpt_path +\
                        '--max_num_dist_1thread %s '%param_dic['max_num_dist_1thread_eval']
    #pdb.set_trace();
    if 0 in tasks: run_cmd(cmd)

    # hist
    hist_fig_path = '%s/hist.norm_%s.png'%(ckpt_dir, param_dic['normalized_hist'])

    cmd = 'python evaluation.py draw_histogram '+\
                     '--pairwise_dist_file %s '%sample_dist +\
                     '--histogram_fig %s '%hist_fig_path +\
                     '--dist_cols %s '%param_dic['dist_cols'] +\
                     '--normalized %s '%param_dic['normalized_hist']
    #pdb.set_trace(); 
    if 1 in tasks: run_cmd(cmd)

    # roc
    roc_fig_path = '%s/roc.norm_%s.png'%(ckpt_dir, param_dic['normalized_roc'])

    cmd = 'python evaluation.py draw_roc '+\
                     '--pairwise_dist_file %s '%sample_dist +\
                     '--roc_fig %s '%roc_fig_path +\
                     '--n_thresholds %s '%param_dic['n_thresholds'] +\
                     '--normalized %s '%param_dic['normalized_roc']
    #pdb.set_trace(); 
    if 2 in tasks: run_cmd(cmd)

    # export embed
    embed_output='%s/sample.embed.fa'

    cmd = 'python evaluation.py export_embedding '+\
                     '--seq_type %s '%param_dic['seq_type'] +\
                     '--input_fa %s '%sample_fa +\
                     '--embed_output %s '%embed_output +\
                     '--model_prefix %s '%ckpt_path
    #pdb.set_trace(); 
    if 3 in tasks: run_cmd(cmd)

    logPrint(logLabel+'End')

    return

'''
input: model_dir, can contain:
                  {ckpt_step_S1_loss_L1.meta or .data or .index}
output:
       ckpt_path: model_dir/ckpt_step_S_loss_L (min L)
       ckpt_name: ckpt_step_S_loss_L
       itms: list of (step, loss, relevant ckpt_name) sorted by step

note:
- it's possible that there's no trained model in model_dir (e.g. loss is nan).
  so ckpt_path is '', ckpt_name is '' and itms is []
'''
def select_ckpt(model_dir):

    #pdb.set_trace()

    filenames = os.listdir(model_dir)
    '''
    example:
    ['checkpoint', 'ckpt_step_14000_loss_0.000184924.meta', 'ckpt_step_15000_loss_0.00024944.data-00000-of-00001', 'ckpt_step_15000_loss_0.00024944.meta', 'ckpt_step_16000_loss_9.85918e-05.data-00000-of-00001', 'ckpt_step_16000_loss_9.85918e-05.index', 'ckpt_step_16000_loss_9.85918e-05.meta', 'ckpt_step_17000_loss_0.000209771.data-00000-of-00001', 'ckpt_step_17000_loss_0.000209771.index', 'ckpt_step_17000_loss_0.000209771.meta', 'ckpt_step_13000_loss_0.000402888.data-00000-of-00001', 'ckpt_step_13000_loss_0.000402888.index', 'ckpt_step_13000_loss_0.000402888.meta', 'ckpt_step_14000_loss_0.000184924.data-00000-of-00001', 'ckpt_step_14000_loss_0.000184924.index', 'ckpt_step_15000_loss_0.00024944.index']
    '''

    filenames = [fn for fn in filenames if len(fn)>=4 and fn[-4:]=='meta']
    #pdb.set_trace()
    '''
    example:
    ['ckpt_step_14000_loss_0.000184924.meta', 'ckpt_step_15000_loss_0.00024944.meta', 'ckpt_step_16000_loss_9.85918e-05.meta', 'ckpt_step_17000_loss_0.000209771.meta', 'ckpt_step_13000_loss_0.000402888.meta']
    '''

    if filenames == []:

        ckpt_path = ''
        ckpt_name = ''
        return ckpt_path, ckpt_name, []

    itms = []
    for fn in filenames:

        tokens = fn[:-5].split('_')
        step = int(tokens[2])
        if tokens[4]=='nan': continue

        loss = float(tokens[4])
        itms.append((step, loss, fn))
    #pdb.set_trace()

    if itms == []:

        ckpt_path = ''
        ckpt_name = ''
        return ckpt_path, ckpt_name, []        

    itms_sort_step = sorted(itms, key=lambda x: x[0])
    itms_sort_loss = sorted(itms, key=lambda x: x[1])

    try:
        ckpt_name = itms_sort_loss[0][2][:-5] #skip .meta
    except:
        pdb.set_trace()
    ckpt_path = model_dir + '/' + ckpt_name

    #pdb.set_trace()

    return ckpt_path, ckpt_name, itms_sort_step

'''
Note:
Parse a config_file (e.g. hdsz=100,200) into multiple eval jobs (e.g. job of hdsz=100, job of hdsz=200)
sequentially eval these jobs (per job is done parallel), and save eval results into locations specified in storage_structure
'''
def batch_test_eval(args):

    logLabel = '[batch_test_eval]'

    logPrint('%s Start'%logLabel)

    config_desc_dic_list = parse_config_file(args.config_file)
    #pdb.set_trace()

    logPrint('%s %s parsed'%(logLabel, args.config_file))

    nJobs = args.N 

    for param_desc, param_dic in config_desc_dic_list:
        batch_test_eval_1job((param_dic, param_desc, args))
    
    logPrint('%s End'%logLabel)

    return

'''
Note:
Parse a config_file,
Generate data (cluster, sample, dist) for train or eval purpose of siamese (enc) architecture
              (seq2seq_cgk) for train of seq2seq architecture
              (siamese_seq2seq) for train or validation of siamese seq2seq architecture
Data stored into locations specified in storage_structure

Note:
- sample.dist is only calculated for train purpose of siamese (enc) architecture
'''

def batch_test_data(args):
    #pdb.set_trace()
    logLabel = '[batch_test_data]'

    logPrint('%s Start'%(logLabel))

    config_desc_dic_list = parse_config_file(args.config_file)
    param_dic = config_desc_dic_list[0][1] #we only need location info from param_dic, param combinations not important here

    #---------- locations
    root_dir = param_dic['root_dir']
    data_label = param_dic['data_label']
    data_dir = '%s/%s/%s/data/'%(root_dir, data_label, args.purpose)

    run_cmd('mkdir -p %s'%data_dir); logPrint('%s created'%data_dir)

    cluster_fa = '%s/cluster.fa'%data_dir
    sample_fa  = '%s/sample.fa'%data_dir
    sample_dist = '%s/sample.dist'%data_dir
    
    seq2seq_pair = '%s/seq2seq_cgk.txt'%data_dir

    #for siamese seq2seq
    #siamese_seq2seq = '%s/siamese_seq2seq.txt'%data_dir

    #---------- what kind of data to generate
    tasks = [int(d) for d in args.tasks.split(',') if d != '']

    #---------- actual commands
    if 0 in tasks:
        cmd = 'python simulate_data.py gen_cluster_center '+\
                        '--output %s '%cluster_fa+\
                        '--seq_type %s '%param_dic['seq_type']+\
                        '--num %s '%param_dic['n_clusters']+\
                        '--length %s '%param_dic['cluster_len']+\
                        '--weight_distr %s '%param_dic['weight_distr']+\
                        '--sid_pre %s '%param_dic['sid_pre']
        run_cmd(cmd)

    if 1 in tasks:
        cmd = 'python simulate_data.py sample_from_cluster '+\
                        '--fa_input %s '%cluster_fa+\
                        '--type %s '%param_dic['seq_type']+\
                        '--fa_output %s '%sample_fa+\
                        '--prefix %s '%param_dic['sample_prefix']+\
                        '--total_samples %s '%param_dic['n_tot_samples']+\
                        '--copy  %s '%param_dic['n_copy']+\
                        '--ins %s '%param_dic['rate_ins']+\
                        '--dele %s '%param_dic['rate_del']+\
                        '--sub %s '%param_dic['rate_sub']+\
                        '--thread %d '%args.N+\
                        '--clear_split_files %s '%param_dic['clear_split']
        run_cmd(cmd)
    
    if 2 in tasks and args.purpose=='train':
        cmd = 'python simulate_data.py calc_dist '+\
                        '--distance_type_list %s '%param_dic['dist_tp_list_data']+\
                        '--seq_type %s '%param_dic['seq_type']+\
                        '--seq_fa %s '%sample_fa+\
                        '--dist_out %s '%sample_dist+\
                        '--thread %d '%args.N+\
                        '--addheader %s '%param_dic['add_hd']+\
                        '--clear_intermediate %s '%param_dic['clear_interm']+\
                        '--max_num_dist_1thread %s '%param_dic['max_num_dist_1thread']
        #pdb.set_trace()
        run_cmd(cmd)

    if 3 in tasks:
        #seq2seq_cgk of seq2seq architecture
        cmd = 'python simulate_data.py gen_seq2seq '+\
                        '--output %s '%seq2seq_pair+\
                        '--seq_type %s '%param_dic['seq_type']+\
                        '--seq2seq_type %s '%param_dic['seq2seq_type']+\
                        '--num %s '%param_dic['n_clusters']+\
                        '--length %s '%param_dic['cluster_len']
        #pdb.set_trace()
        run_cmd(cmd)

        if args.purpose=='train' and param_dic['n_clusters_validation']>0:
            #generate validation data
            seq2seq_pair_validation = '%s/seq2seq_cgk_validation.txt'%data_dir
            cmd = 'python simulate_data.py gen_seq2seq '+\
                            '--output %s '%seq2seq_pair_validation+\
                            '--seq_type %s '%param_dic['seq_type']+\
                            '--seq2seq_type %s '%param_dic['seq2seq_type']+\
                            '--num %s '%param_dic['n_clusters_validation']+\
                            '--length %s '%param_dic['cluster_len']
            #pdb.set_trace()
            run_cmd(cmd)

    if 4 in tasks:


        siamese_seq2seq = '%s/siamese_seq2seq.txt'%data_dir
        
        #siamese_seq2seq of siamese seq2seq architecture
        cmd = 'python simulate_data.py gen_siamese_seq2seq '+\
                        '--output %s '%siamese_seq2seq+\
                        '--seq_type %s '%param_dic['seq_type']+\
                        '--si_correlation_type %s '%param_dic['si_correlation_type']+\
                        '--num %s '%param_dic['n_clusters']+\
                        '--length %s '%param_dic['cluster_len']+\
                        '--length2 %s '%param_dic['cluster_len2']+\
                        '--rate_ins %s '%param_dic['rate_ins']+\
                        '--rate_del %s '%param_dic['rate_del']+\
                        '--rate_sub %s '%param_dic['rate_sub']
        #pdb.set_trace()
        run_cmd(cmd)

        if args.purpose=='train' and param_dic['n_clusters_validation']>0:
            #generate validation data
            siamese_seq2seq_validation = '%s/siamese_seq2seq_validation.txt'%data_dir
            cmd = 'python simulate_data.py gen_siamese_seq2seq '+\
                            '--output %s '%siamese_seq2seq_validation+\
                            '--seq_type %s '%param_dic['seq_type']+\
                            '--si_correlation_type %s '%param_dic['si_correlation_type']+\
                            '--num %s '%param_dic['n_clusters_validation']+\
                            '--length %s '%param_dic['cluster_len']+\
                            '--length2 %s '%param_dic['cluster_len2']+\
                            '--rate_ins %s '%param_dic['rate_ins']+\
                            '--rate_del %s '%param_dic['rate_del']+\
                            '--rate_sub %s '%param_dic['rate_sub']
            #pdb.set_trace()
            run_cmd(cmd)

    logPrint('%s End'%(logLabel))
    #
    
    return

'''
main
'''
if __name__ == "__main__":

    parser = ArgumentParser()
    subs = parser.add_subparsers()

    if sys.argv[1]=='data':
        s_parser = subs.add_parser('data')

        s_parser.add_argument('--config_file', type=str, help='config file (locations and params)')
        s_parser.add_argument('--purpose', type=str, help='train or eval; or validation (same location as train)')
        s_parser.add_argument('--tasks', type=str, help='tp1[,tp2...]; tp=0 cluster 1 sample 2 dist (for train purpose)'+\
                                                                      'tp=3 seq2seq (cgk etc to be configured)'+\
                                                                      'tp=4 siamese_seq2seq; order does not matter')
        s_parser.add_argument('--N', type=int, default=1, help='number of processes to use; default 1')

        args = parser.parse_args()

        #pdb.set_trace()
        batch_test_data(args)

    elif sys.argv[1]=='train':
        s_parser = subs.add_parser('train')

        s_parser.add_argument('--config_file', type=str, help='config file (locations and params)')
        s_parser.add_argument('--N', type=int, default=1, help='number of processes to use; default 1')

        s_parser.add_argument('--a', type=int, default=0, help='type of architecture, 0=siamese (default) and 1=seq2seq and 2=siamese_seq2seq')
        args = parser.parse_args()

        batch_test_train(args)

        #pdb.set_trace()

    elif sys.argv[1]=='eval':

        s_parser = subs.add_parser('eval')

        s_parser.add_argument('--config_file', type=str, help='config file (locations and params)')
        s_parser.add_argument('--tasks', type=str, help='tp1[,tp2...]; tp=0 calc nn etc dist 1 hist 2 roc 3 export embedding; order does not matter')
        s_parser.add_argument('--N', type=int, default=1, help='number of processes to use; default 1')

        args = parser.parse_args()

        batch_test_eval(args)

        pdb.set_trace()

    elif sys.argv[1]=='parse_config':
        s_parser = subs.add_parser('parse_config')

        s_parser.add_argument('--config_file', type=str, help='config file (locations and params)')

        args = parser.parse_args()

        config_desc_dic_list = parse_config_file(args.config_file)

        pdb.set_trace()

    elif sys.argv[1]=='select_ckpt':
        s_parser = subs.add_parser('select_ckpt')

        s_parser.add_argument('--model_dir', type=str, help='model_dir')

        args = parser.parse_args()

        select_ckpt(args.model_dir)

        pdb.set_trace()
    
