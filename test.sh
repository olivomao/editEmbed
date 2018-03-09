#[1] Generate clusters

python simulate_data.py gen_cluster_center --output data_test/bin.fa \
                                           --seq_type 0 --num 100 --length 150 --weight_distr 0

#python simulate_data.py gen_cluster_center --output data_test/dna.fa \
#                                           --seq_type 1 --num 5 --length 10 --weight_distr 1 --sid_pre cc

#[2] Sample noisy seqs -- indel_channel

#python indel_channel.py data_test/bin.fa s --total_samples 100 --copy -1 --ins 0 --dele 0 --sub 0 \
#                        --output data_test/bin_noisy.fa --type 0

#python indel_channel.py data_test/dna.fa s --total_samples 100 --copy -1 --ins 0 --dele 0 --sub 0.2 \
#                        --output data_test/dna_noisy.fa --type 1

#[3] Sample noisy seqs -- wrap up

python simulate_data.py sample_from_cluster 	 --fa_input data_test/bin.fa \
	                                             --type 0 \
	                                             --fa_output data_test/bin_noisy_copy10.fa \
	                                             --prefix s \
	                                             --total_samples -1 \
	                                             --copy 10 \
	                                             --ins 0.07 \
	                                             --dele 0.07 \
	                                             --sub 0.01 \
	                                             --thread 10 \
	                                             --clear_split_files 1

#python simulate_data.py sample_from_cluster 	 --fa_input data_test/dna.fa \
#	                                             --type 1 \
#	                                             --fa_output data_test/dna_noisy_thread2_tot100.fa \
#	                                             --prefix sample \
#	                                             --total_samples 100 \
#	                                             --copy -1 \
#	                                             --ins 0 \
#	                                             --dele 0 \
#	                                             --sub 0.1 \
#	                                             --thread 2 \
#	                                             --clear_split_files 1

#[4] Calc distance -- one thread

#python simulate_data.py calc_dist_1thread  --distance_type     0 \
#                                           --seq_type          0 \
#                                           --seq_fa1           data_test/bin_noisy_thread2_copy2.fa \
#                                           --seq_fa2           data_test/bin_noisy_thread2_copy2.fa \
#                                           --dist_out          data_test/bin_noisy_thread2_copy2.dist \
#                                           --addheader         0

#python simulate_data.py calc_dist_1thread  --distance_type     0 \
#                                           --seq_type          1 \
#                                           --seq_fa1           data_test/dna_noisy_thread1_copy2.fa \
#                                           --seq_fa2           data_test/dna_noisy_thread1_copy2.fa \
#                                           --dist_out          data_test/dna_noisy_thread1_copy2.dist \
#                                           --addheader         1

#[5] Calc distance -- multiple threads

python simulate_data.py calc_dist --distance_type  0 \
                                  --seq_type       0 \
                                  --seq_fa         data_test/bin_noisy_copy10.fa \
                                  --dist_out       data_test/bin_noisy_copy10.dist \
                                  --thread         10 \
                                  --addheader      0 \
                                  --clear_intermediate 1