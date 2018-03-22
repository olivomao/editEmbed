import time, subprocess, sys, pdb, os, multiprocessing

def run_cmd(cmd, shell=True):

    subprocess.call(cmd, shell=shell)

def run_cmds(cmds,noJobs=20):

    if noJobs==1:
        #pdb.set_trace()
        for cmd in cmds:
            run_cmd(cmd)
    else:
        #cmds is a tuple of strings
        #noJobs is a an integer with the no of Jobs to run in parallel
        p = multiprocessing.Pool(noJobs)
        p.map(run_cmd,cmds)

'''
print [asctime] log_msg on the screen
'''
def logPrint(log_msg):

    st = '[%s] %s'%(str(time.asctime()), log_msg)

    print(st+'\n')

    return st

'''
return parent_dir and filename of abs dir_path
'''
def parent_dir(dir_path):
    if dir_path[0]=='/':
        isRelative = False 
    else:
        isRelative = True
    dir_path = dir_path.split('/')
    dir_path = [itm for itm in dir_path if itm != '']
    
    #pdb.set_trace()
    if isRelative==False:
        return '/'+'/'.join(dir_path[:-1])+'/', dir_path[-1]
    else:
        if len(dir_path)==1:
            return '', dir_path[-1]
        else:
            return '/'.join(dir_path[:-1])+'/', dir_path[-1]

'''
split fa_path (w/ fn.txt) into out_dir (w/ fn's path as default)/fn_1.txt,fn_2.txt,...,fn_M.txt each w/ num_seq_per_file seqs
'''
def split_fa_file(fa_path, out_dir="", num_seq_per_file=20):

    #pdb.set_trace()

    path, fn = os.path.split(fa_path)
    if out_dir=="": out_dir = path 

    with open(fa_path, 'r') as r:
        name, ext = os.path.splitext(fn)
        try:
            j = 0; cnt_seq=0;
            #filename = os.path.join(out_dir, '{}_{}{}'.format(name, j, ext))
            filename = os.path.join(out_dir, '%s_%05d%s'%(name, j, ext))
            w = open(filename, 'w')
            for i,line in enumerate(r):
                if line == '':
                    continue
                elif line[0] == '>':
                    if i==0:
                        cnt_seq = 1

                    elif cnt_seq%num_seq_per_file==0:
                        w.close(); print('%s written'%filename)
                        j+=1
                        #filename = os.path.join(out_dir, '{}_{}{}'.format(name, j, ext))
                        filename = os.path.join(out_dir, '%s_%05d%s'%(name, j, ext))
                        w = open(filename, 'w')
                        cnt_seq = 1

                    else:
                        cnt_seq += 1
                w.write(line)
        finally:
            w.close(); print('%s written'%filename)
    return


'''
merge files in in_dir into file_path
if dele is True, all files in in_dir will be deleted

if headerline != '' (need '\n'), add headerline at beginning
'''
def merge_files(in_dir, file_path, dele=False, headerline=''):

    targets = os.listdir(in_dir)
    targets.sort()

    #pdb.set_trace()

    with open(file_path, 'w') as fo:
        if headerline != '':
            fo.write(headerline)

        for target in targets:
            target_path = in_dir+'/'+target
            with open(target_path, 'r') as fi:
                for line in fi:
                    fo.write(line)

            if dele == True:
                cmd = 'rm %s'%target_path
                run_cmd(cmd)
    return

'''
used to show progress on screen
''' 
class iterCounter:

    def __init__(self, N, msg):

        self.N = N
        self.msg = msg
        self.T = N/100
        self.p = 0
        self.q = 0

        return

    def finish(self):

        print('')#print an empty line for display purpose

        return

    def inc(self):

        if self.N==0:
            return

        if self.N<100:
            self.p += 1
            sys.stdout.write('\r');
            sys.stdout.write('[%s] %s: %.2f %%'%(str(time.asctime()), self.msg, float(self.p)*100.0/self.N));
            sys.stdout.flush()

        else:
            self.p += 1
            if self.p >= self.T:
                self.p = 0
                self.q += 1
                sys.stdout.write('\r');
                sys.stdout.write('[%s] %s: %d %%'%(str(time.asctime()), self.msg, self.q));
                sys.stdout.flush()

        return  