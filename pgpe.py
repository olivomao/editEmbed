import numpy as np
import tensorflow as tf
import pdb

class BasePGPE(object):

    def __init__(self,
                 sess,
                 N,             #update meta param \rho per N batches
                 learning_rate, #e.g. beta in config file
                 ):
        print('BasePGPE init')

        self.vs = self.init_vs(sess) 

        self.N = N #update meta param \rho (=u and s in vs) per N batches
        self.learning_rate = learning_rate

        return

    '''
    v is i-th theta param from nn model
    u,s are i-th meta param controlling v
    '''
    def calc_du(self, v, u, s):

        d_u = (v - u)/np.square(s)
        ## for numerical stability
        nan_indice = np.isnan(d_u)
        d_u[nan_indice] = 0.0
        inf_indice = np.isinf(d_u)
        d_u[inf_indice] = 0.0

        return d_u

    def calc_ds(self, v, u, s):

        d_s = (np.square(v-u)-np.square(s))/(s*np.square(s))
        ## for numerical stability
        nan_indice = np.isnan(d_s)
        d_s[nan_indice] = 0.0
        inf_indice = np.isinf(d_s)
        d_s[inf_indice] = 0.0

        return d_s

    '''
    PGPE update for i-th meta param (u or s)
    u or s is denoted as x
    
    The update is via gradient ascent
    across N=len(r_list) batches
    among these N batches, meta param not changed,
    but network param changed per batch

    note: s needs to be non-negative
    '''
    def grad(self, x, dx_list, r_list, ge0=False):
        
        D_x = dx_list[0]*r_list[0]
        for n in xrange(1,len(r_list)):
            D_x += dx_list[n]*r_list[n]

        x = x + self.learning_rate*D_x
        x = 1.0/float(self.N)*x
        if ge0==True:
            x[x<0]=0 #0.00001
            #x = np.abs(x)
        return x

    '''
    rand init weights into the model

    vc_dic is returned with key=i-th Variable theta(i)[n] ~ N(u(i), sigma(i))
                            val={'u':u(i),
                                 's':sigma(i),
                                 'du': [...du(i)[n]...],
                                 'ds': [...ds(i)[n]...],
                                 'r':  [....r[n]....],
                                } 
    n represents 0~N-1-th batch in order to update i-th meta params
    u(i) and sigma(i)
    '''
    def init_vs(self, sess):

        vs_dic = {}

        for v in tf.trainable_variables():
            vs_dic[v] = {}
            sp = v.eval().shape

            #update u, s
            u = v.eval() #from pre-trained model
            vs_dic[v]['u']=u
            s = np.random.uniform(0, 0.001, size=sp) #sigma_v
            vs_dic[v]['s']=s

            #sample new_v
            new_v = np.random.normal(loc=u, scale=s)
            v.load(new_v, sess)

            #to be used for next update of u,s
            vs_dic[v]['du']=[]
            vs_dic[v]['du'].append(self.calc_du(new_v, u, s))
            vs_dic[v]['ds']=[]
            vs_dic[v]['ds'].append(self.calc_ds(new_v, u, s))
            vs_dic[v]['r']=[]

        return vs_dic

    def update(self, sess, metrics):

        #n-th batch rewards
        rewards = - np.abs(metrics.dev_de_d_nn)
        avg_rewards_per_batch = np.mean(rewards)

        for v, cdic in self.vs.items():

            self.vs[v]['r'].append(avg_rewards_per_batch)

            if len(self.vs[v]['r'])==self.N: #update u,s using du,ds and r
                print('PGPE update meta param...')
                #pdb.set_trace()
                u = cdic['u']
                s = cdic['s']

                du_list = cdic['du']
                ds_list = cdic['ds']
                r_list  = cdic['r']
                cdic['u'] = self.grad(u, du_list, r_list)
                cdic['s'] = self.grad(s, ds_list, r_list, ge0=True)
                
                cdic['du']=[]
                cdic['ds']=[]
                cdic['r']=[]

        for v, cdic in self.vs.items():
            u = cdic['u']
            s = cdic['s']

            #sample new_v
            new_v = np.random.normal(loc=u, scale=s)
            v.load(new_v, sess)

            #to be used for next update of u,s
            cdic['du'].append(self.calc_du(new_v, u, s))
            cdic['ds'].append(self.calc_ds(new_v, u, s))

        return


