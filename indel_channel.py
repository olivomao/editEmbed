import sys, random

simType = ['sub', 'ins', 'del', 'match']
simTypeSize = 4

def throwdice(profile):
    dice = random.random()
    for i in xrange(simTypeSize):
        if dice < profile[i]:
            return simType[i]

'''
simulate a binary seq through a long read channel

input:
seq - input binary seq
sub - sub rate
ins - ins rate
dele - dele rate
(need: 0 <= sub + ins + del < 1)
'''
def sim_seq_binary(seq, sub, dele, ins):

    if(sub+ins+dele>=1):
        print >> sys.stderr, "Total sub+ins+del error can not exceed 1!"
        sys.exit(-1)

    profile = [sub, sub+ins, sub+ins+dele, 1.]

    nucl = set(['0', '1'])
    sim = ''
    for i, s in enumerate(seq):
        while True:
            tp = throwdice(profile)
            if tp=='match': 
                sim += s
                break
            elif tp=='ins':
                # insertion occurred, with 1/4 chance it'll match
                choice = random.sample(nucl,1)[0]
                sim += choice
            elif tp=='sub': # anything but the right one
                choice = random.sample(nucl.difference([s]),1)[0]
                sim += choice
                break
            elif tp=='del': # skip over this
                break
            else: raise KeyError, "Invalid type {0}".format(tp)

    return sim