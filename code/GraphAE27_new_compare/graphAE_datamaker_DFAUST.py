from os import mkdir
from os.path import join, exists
import h5py
import numpy as np

def get_pcs_from_DFAUST(fname):
    n=0
    pcs =[]
    with h5py.File(fname,'r') as f:
        for sidseq in f:
            if(sidseq!='faces'):
                verts = f[sidseq].value.transpose([2, 0, 1])
                # Write to an obj file
                for iv, v in enumerate(verts):
                    pcs += [v]
                    n=n+1
                    if(n%1000 ==0):
                        print ("load", n, "models")
    
    return pcs

def split_train_and_eval_and_test_data(pcs, train_fn, eval_fn, test_fn):
    
    np.random.shuffle(pcs)
    num = pcs.shape[0]
    train_pcs =pcs[0:int(num*0.8)]
    eval_pcs = pcs[int(num*0.8):int(num*0.9)]
    test_pcs = pcs[int(num*0.9):]
    
    
    np.save(train_fn, train_pcs)
    np.save(eval_fn,  eval_pcs)
    np.save(test_fn,  test_pcs)


pcs1 = get_pcs_from_DFAUST("/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/body/data/D-FAUST/registrations_f.hdf5")
pcs2 = get_pcs_from_DFAUST("/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/body/data/D-FAUST/registrations_m.hdf5")

pcs = pcs1 + pcs2
pcs = np.array(pcs)

print ("Load", pcs.shape[0], "models in total.")

train_fn ="../../data/D-FAUST/train"
eval_fn ="../../data/D-FAUST/eval"
test_fn ="../../data/D-FAUST/test"

mean_fn = "../../data/D-FAUST/mean"
std_fn = "../../data/D-FAUST/std"

split_train_and_eval_and_test_data(pcs, train_fn, eval_fn, test_fn)

pcs_mean  = pcs.mean(0)
pcs_std = pcs.std(0)

np.save(mean_fn, pcs_mean)
np.save(std_fn, pcs_std)






