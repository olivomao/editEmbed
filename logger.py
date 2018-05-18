import numpy as np
import os
import abc
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Logger(object):

    def __init__(self, path):

        self.path = path

        self.log_file = open(path, 'w')

        return

    def close(self):
        
        if self.log_file.closed == False:
            print('%s written'%self.path)
            self.log_file.close()

    '''
    add_log(col_name=col_vals,...)
    '''
    @abc.abstractmethod
    def add_log(self, **kwargs):
        pass

    '''
    log_file should to be closed, and will reopen
    '''
    @abc.abstractmethod
    def plot(self, **kwargs):
        pass

    def add_log_using_keys(self, kwargs, keys):
        
        lengths = [len(kwargs[k]) for k in keys]
        L = lengths[0]
       
        assert min(lengths)==L and max(lengths)==L

        for i in range(L):
            st = '\t'.join(str(kwargs[k][i]) for k in keys) + '\n'
            self.log_file.write(st)

        return



class DevLogger(Logger):

    def __init__(self, path):
        super(DevLogger, self).__init__(path)

    #k,v in kwargs.items() here v can be scaler or vector
    #usage: add_log(batch_index=..., dev_cgk=..., dev_nn=..., s_i_type=...)
    def add_log(self, **kwargs):

        keys = ['batch_index', 'dev_cgk', 'dev_nn', 's_i_type']

        self.add_log_using_keys(kwargs, keys)

    '''
    plot two subplots dev_cgk and dev_nn,
         per subplot two curves wrt s_i_type 0/1
    
    kwargs:
    - nbins=number of bins for histogram, default 20
    - output_fig: default self.path+'.png'
    '''
    def plot(self, **kwargs):

        assert self.log_file.closed == True

        if 'nbins' in kwargs:
            nbins = int(kwargs['nbins'])
        else:
            nbins = 20

        if 'output_fig' in kwargs:
            output_fig = kwargs['output_fig']
        else:
            output_fig = self.path + '.png'

        print('DevLogger plot of %s'%self.path)

        dv_cgk = {0:[],1:[]} #subplot dv_cgk has two curves wrt s_i_type 0/1
        dv_nn  = {0:[],1:[]}
        min_x  = None
        max_x  = None
        with open(self.path, 'r') as f:

            for line in f:

                if line[0]=='#': continue
                tokens = [t for t in line.split() if t!='']
                if len(tokens)<4: continue

                c = float(tokens[1])
                n = float(tokens[2])
                if min_x is None:
                    min_x = min(c,n)
                    max_x = max(c,n)
                else:
                    min_x = min(min_x, min(c,n))
                    max_x = max(max_x, max(c,n))
                s = int(tokens[3])

                dv_cgk[s].append(c)
                dv_nn[s].append(n)

        pdb.set_trace()
        fig, axes = plt.subplots(2)

        r_min = min_x; r_max = max_x
        rg = np.arange(r_min-r_min/2.0, r_max+r_max/2.0, (r_max-r_min)/nbins)

        subplot_list = [[dv_cgk, 'dv_cgk'], [dv_nn,'dv_nn']]
        for i_subplot in range(len(subplot_list)):
            
            dv_dic = subplot_list[i_subplot][0]
            dv_lab = subplot_list[i_subplot][1]

            for k in dv_dic.keys():
                dev_vals = dv_dic[k]

                if dev_vals==[]:
                    print('%s[%d] has no vals'%(dv_lab, k))
                    continue

                try:
                    h, v = np.histogram(dev_vals, bins=rg)
                    axes[i_subplot].plot(v[1:], h, marker='o',label='%s-%d'%(dv_lab,k))
                except:
                    print('exception')
                    pdb.set_trace()
            
            axes[i_subplot].legend()

        plt.tight_layout()
        plt.savefig(output_fig)

        print('DevLogger draws %s'%output_fig)

        return


















        


