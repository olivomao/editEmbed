alias BEGINCOMMENT="if [ ]; then"
alias ENDCOMMENT="fi"

Python=/usr/bin/python

num_seq=10000
seq_len=100
insertion_rate=0.01
deletion_rate=0.01
substitution_rate=0.01
distance_metric=0
train_data=data_sim_binary/data.txt

BEGINCOMMENT
python simulate_binary.py -N $num_seq \
                          -L $seq_len \
                          -i $insertion_rate \
                          -d $deletion_rate \
                          -s $substitution_rate \
                          -D $distance_metric \
                          -o $train_data
ENDCOMMENT

train_dir=data_sim_binary/
bs=100
n_epochs=500

maxlen=500
blocklen=10
emd_sz=200
n_layers=3
hd_sz=100
lrate=0.001
dpt=1.0
#for ksreeram server: $Python train.py use_simulate_binary \
python train.py use_simulate_binary \
                --input_type 0 \
                --train_input $train_data \
                --seq_type 0 \
                --train_output_dir $train_dir \
                --batch_size $bs \
                --num_epochs $n_epochs \
                --maxlen  $maxlen \
                --blocklen  $blocklen \
                --embedding_size $emd_sz \
                --num_layers $n_layers \
                --hidden_sz $hd_sz \
                --learning_rate $lrate \
                --dropout $dpt
