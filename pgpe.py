import numpy as np
import tensorflow as tf
import pdb

class BasePGPE(object):

    def __init__(self,
                 sess,
                 N,             #update meta param \rho per N batches
                 learning_rate, #e.g. beta in config file
                 init_sigma_range,
                 ):
        print('BasePGPE init')

        #0 basic version; 
        #1 with Algo1 basic version
        #2 with Algo2 symmetric sampling
        self.method = 1 
        self.baseline = None
        self.vs = self.init_vs(sess) 

        self.N = N #update meta param \rho (=u and s in vs) per N batches
        self.learning_rate = learning_rate
        self.init_sigma_range = init_sigma_range

        return

    '''
    v is i-th theta param from nn model
    u,s are i-th meta param controlling v
    '''
    def calc_du(self, v, u, s):

        if self.method==0:
            d_u = (v - u)/np.square(s)
        elif self.method==1:
            d_u = v - u
        ## for numerical stability
        nan_indice = np.isnan(d_u)
        d_u[nan_indice] = 0.0
        inf_indice = np.isinf(d_u)
        d_u[inf_indice] = 0.0

        return d_u

    def calc_ds(self, v, u, s):

        if self.method==0:
            d_s = (np.square(v-u)-np.square(s))/(s*np.square(s))
        elif self.method==1:
            d_s = (np.square(v-u)-np.square(s))/s
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

        if ge0 == True:
            print('Before grad update, min=%f, max=%f'%(np.min(x), np.max(x)))

        x = x + self.learning_rate*D_x

        if ge0 == True:
            m = np.min(x)
            M = np.max(x)
            if m<0 or M<0:
                flag_st = '*'
                pdb.set_trace()
            else:
                flag_st = ''
            print('After grad update, min=%f, max=%f learning_rate=%f %s'%(m, M, self.learning_rate, flag_st))

        #x = 1.0/float(self.N)*x
        if ge0==True:
            #x[x<0]=0 #0.00001
            x = np.abs(x)
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
            s = np.random.uniform(0, self.init_sigma_range, size=sp) #sigma_v
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

    '''
    method==1 basic version

    return baseline (rewards, could be None)
    to see if policy update is working reasonably
    '''
    def update(self, sess, metrics):

        #n-th batch rewards
        rewards = - np.abs(metrics.dev_de_d_nn)
        avg_rewards_per_batch = np.mean(rewards)

        if self.baseline is None:
            self.baseline = avg_rewards_per_batch
        else:
            self.baseline = 0.9*self.baseline + 0.1*avg_rewards_per_batch

        if self.method==1:
            avg_rewards_per_batch = avg_rewards_per_batch - self.baseline

        for v, cdic in self.vs.items():

            self.vs[v]['r'].append(avg_rewards_per_batch)

            if len(self.vs[v]['r'])==self.N: #update u,s using du,ds and r
                print('PGPE update meta param...')
                u = cdic['u']
                s = cdic['s']

                du_list = cdic['du']
                ds_list = cdic['ds']
                r_list  = cdic['r']
                print(v)
                cdic['u'] = self.grad(u, du_list, r_list)
                cdic['s'] = self.grad(s, ds_list, r_list, ge0=True)
                print('')
                
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

        if self.baseline is None:
            reward_str = ''
        else:
            reward_str = str(self.baseline)
        return self.baseline, reward_str


