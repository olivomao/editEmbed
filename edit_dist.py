'''
module of calc seq of a/b (binary seq or dna seq)

we could just use existing resources. [1] and [2] perform similar according to [1]'s benchmark

[1] https://pypi.python.org/pypi/editdistance

import editdistance
editdistance.eval('banana', 'bahama')

output: edit distance in long int

[2] https://github.com/ztane/python-Levenshtein/ #from [1]

import Levenshtein
Levenshtein.distance('010','010')

output: edit distance in int

'''

#import Levenshtein
import editdistance
import affinegap as ag
import pdb

def edit_dist(a,b):
	#return float(Levenshtein.distance(a,b))/(0.5*(len(a)+len(b)))
	#return float(editdistance.eval(a,b))/(0.5*(len(a)+len(b)))
	return float(editdistance.eval(a,b))/(len(a)+len(b))
	#return editdistance.eval(a,b)

def gapped_edit_dist(seq1_str, seq2_str):
	
	d1 = ag.normalizedAffineGapDistance(seq1_str, seq2_str,
		matchWeight=0, mismatchWeight=1,gapWeight=1,spaceWeight=5,
		abbreviation_scale=1)

	return d1