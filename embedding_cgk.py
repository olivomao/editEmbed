from global_vals import *
from proc_data import *

'''
Base2Index = {'A':0, 'C':1, 'T':2, 'G':3}

def cgk_embedding(s, N):

    #pdb.set_trace()

    s = s.upper()

    L_R = 3*N*4
    R = np.random.randint(0,2,L_R)
    
    i = 0
    s1 = ""
    for j in range(3*N):
        if i<len(s):
            c = s[i]
            s1 = s1 + c
            i = i + R[(j-1)*4+Base2Index[c]]
        else:
            s1 = s1 + "N"
    return s1
'''

'''
s: length N
R: length 3N*2
'''

Base2Index = {'0':0, '1':1}

# s is zero padded
def cgk_embedding_binary(s, R):

    #L_R = 3*N*2
    #R = np.random.randint(0,2,L_R)

    N = len(s)
    
    i = 0
    s1 = ""
    for j in range(3*N):
        if i<len(s):
            c = s[i]
            s1 = s1 + c
            i = i + R[(j-1)*2+Base2Index[c]]
        else:
            s1 = s1 + "0"
    return s1

def hamming(a,b):
    return sum([1 for i in range(len(a)) if a[i]!=b[i]])

def draw_histogram(val_list, path):

    import matplotlib.pyplot as plt 

    plt.hist(val_list, normed=True, bins=30)

    plt.savefig(path) #plt.show()

    print('fig saved at %s'%path)

    #pdb.set_trace()

    return

if __name__ == "__main__":

    L_R = 3*MAXLEN*2

    R_list = [np.random.randint(0, 2, L_R) for _ in range(10)] #10 rand seq

    dir = 'data/case_1517522797/'

    eval_file = 'data/eval.0.2.txt'

    fig_path = dir + '/fig_cgk.0.2.png'

    x1, x2, y, x1_str, x2_str = load(eval_file) #deprecated

    diff_list = []

    pdb.set_trace()

    for i in range(len(x1_str)):

        hamming_i = MAXLEN*2

        a = x1_str[i]
        a = a + '0'*(MAXLEN-len(a))

        b = x2_str[i]
        b = b + '0'*(MAXLEN-len(b))

        for R in R_list:

            x1_cgk = cgk_embedding_binary(a, R)

            x2_cgk = cgk_embedding_binary(b, R)

            hamming_i = min(hamming_i, hamming(x1_cgk, x2_cgk))

        if y[i]==0: continue

        diff = float(hamming_i - y[i]) / y[i]

        #print('x1=%s, x2=%s, y=%d, cgk_hamming=%d, diff=%f'%(a, b, y[i], hamming_i, diff))

        #pdb.set_trace()

        diff_list.append(diff)

    draw_histogram(diff_list, fig_path)







