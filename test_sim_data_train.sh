alias BEGINCOMMENT="if [ ]; then"
alias ENDCOMMENT="fi"

Python=/usr/bin/python #TensorFlow supported

#I/O
dir=data_sim_data_type_dna/train/
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
sample_fa=$dir/sample_train_rate_001_to_003.fa
#sample config
sample_prefix=sp_tr_r003
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
dist_tp_list=0 #0:edit 3:nn_dist
sample_dist=$dir/sample_train_rate_001_to_003.dist
add_hd=1
clear_interm=1
model_prfx=$dir/model/ckpt #model_ksreeram_server/ckpt
max_num_dist_1thread=-1
#ENDCOMMENT

BEGINCOMMENT
dist_tp_list=3
sample_dist=$dir/sample_edit_thread20_max-1.dist
add_hd=1
clear_interm=1
model_prfx=$dir/model/ckpt
max_num_dist_1thread=10
ENDCOMMENT

BEGINCOMMENT
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
ENDCOMMENT

#training data
train_output_dir=$dir/model/
max_num_to_sample=-1 #in case too many (inbalanced) dist pairs
batch_sz=100
n_epochs=500
#model params
load_model=0 #to resume a previously trained model (1) or not (0)
maxlen=500
blocklen=10
embd_sz=200
n_layers=3
hidden_sz=100
#opt and regularization
lrate=0.001
dropout=1.0

#[4] Based on sample_fa and sample_dist, prepare training data and do training

#BEGINCOMMENT
#$Python train.py use_simulate_data \
python train.py use_simulate_data \
                 --input_type                  1 \
                 --train_input1                $sample_fa \
                 --train_input2                $sample_dist \
                 --seq_type                    $seq_tp \
                 --train_output_dir            $train_output_dir \
                 --max_num_to_sample           $max_num_to_sample \
                 --batch_size                  $batch_sz \
                 --num_epochs                  $n_epochs \
                 --load_model                  $load_model \
                 --maxlen                      $maxlen \
                 --blocklen                    $blocklen \
                 --embedding_size              $embd_sz \
                 --num_layers                  $n_layers \
                 --hidden_sz                   $hidden_sz \
                 --learning_rate               $lrate \
                 --dropout                     $dropout  
#ENDCOMMENT     

#[5] Evaluate -- calc dist             
                   

