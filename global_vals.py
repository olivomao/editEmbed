MAXLEN = 200 #max len of input seqs; zero padding if smaller
BLOCKLEN = 10 #per seq chopped into MAXLEN/BLOCKLEN blocks, per block is a word to be embedded
BLOCKS = MAXLEN/BLOCKLEN
NUMEMBED = 1024 #2^BLOCKLEN