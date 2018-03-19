alias BEGINCOMMENT="if [ ]; then"
alias ENDCOMMENT="fi"

Python=/usr/bin/python #TensorFlow supported

#I/O
dir=data_sim_data_type_dna/eval/
train_dir=data_sim_data_type_dna/train/
mkdir -p $dir 

cluster_fa=$dir/cluster.fa
#clusters
seq_tp=1
cluster_len=150
weight_dist=0
n_clusters=100
sid_pre=cc #prefix for clusers

#[1] Generate clusters
BEGINCOMMENT
python simulate_data.py gen_cluster_center \
                        --output $cluster_fa \
                        --seq_type $seq_tp \
                        --num $n_clusters \
                        --length $cluster_len \
                        --weight_distr $weight_dist \
                        --sid_pre $sid_pre
ENDCOMMENT

#I/O
sample_fa=$dir/sample_rate_001_to_003.fa
#sample config
sample_prefix=sp_ev_r003
n_tot_samples=-1 #if >0 then each cluster has n_tot_samples*weight seqs to be sampled
n_copy=10
#channel config
rate_ins=0.03
rate_del=0.03
rate_sub=0.03
#parallel config
n_threads=40
clear_split=1 #only used for multi-threads

#[2] Sample noisy seqs

BEGINCOMMENT
python simulate_data.py sample_from_cluster \
                        --fa_input $cluster_fa \
                        --type $seq_tp \
                        --fa_output $sample_fa \
	                --prefix $sample_prefix \
                        --total_samples $n_tot_samples \
	                --copy $n_copy \
	                --ins $rate_ins \
	                --dele $rate_del \
	                --sub $rate_sub \
	                --thread $n_threads \
	                --clear_split_files $clear_split
ENDCOMMENT

#[3] Calc distance -- multiple threads

#BEGINCOMMENT
#load model for eval
dist_tp_list=0,3 #0:edit 3:nn_dist
sample_dist=$dir/sample_rate_001_to_003.dist
add_hd=1
clear_interm=1
model_prfx=$train_dir/model/ckpt #model_ksreeram_server/ckpt
max_num_dist_1thread=-1
#ENDCOMMENT

#BEGINCOMMENT
python simulate_data.py calc_dist \
                        --distance_type_list   $dist_tp_list \
                        --seq_type             $seq_tp \
                        --seq_fa               $sample_fa \
                        --dist_out             $sample_dist \
                        --thread               $n_threads \
                        --addheader            $add_hd \
                        --clear_intermediate   $clear_interm \
                        --model_prefix         $model_prfx \
                        --max_num_dist_1thread $max_num_dist_1thread
#ENDCOMMENT

BEGINCOMMENT

use_normalization=1
histogram_fig=$dir/sample_edit_nn.hist.png

python evaluation.py draw_histogram \
                     --pairwise_dist_file $sample_dist \
                     --histogram_fig $histogram_fig \
                     --normalized $use_normalization

roc_fig=$dir/sample_edit_nn.roc.png 
n_thresholds=100

python evaluation.py draw_roc \
                     --pairwise_dist_file $sample_dist \
                     --roc_fig $roc_fig \
                     --n_thresholds $n_thresholds \
                     --normalized $use_normalization

ENDCOMMENT
         
                   

