import time, subprocess, sys, pdb

def run_cmd(cmd, shell=True):

    subprocess.call(cmd, shell=shell)

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