from os import mkdir
from os.path import join, exists
import h5py
import numpy as np

def get_pc_seqs_from_DFAUST(fname):
    n=0
    pc_seqs =[]
    with h5py.File(fname,'r') as f:
        for sidseq in f:
            if(sidseq!='faces'):
                verts = f[sidseq].value.transpose([2, 0, 1])
                # Write to an obj file
                pc_seq = []
                for iv, v in enumerate(verts):
                    pc_seq += [v]
                    n=n+1
                    if(n%1000 ==0):
                        print ("load", n, "models")
                pc_seqs+=[pc_seq]

    return pc_seqs

def split_train_and_eval_and_test_data(pc_seqs, train_fn, eval_fn, test_fn):
    
    np.random.shuffle(pc_seqs)
    num = len(pc_seqs)
    train_pc_seqs =pc_seqs[0:int(num*0.8)]
    eval_pc_seqs = pc_seqs[int(num*0.8):int(num*0.9)]
    test_pc_seqs = pc_seqs[int(num*0.9):]
    
    print ("train eval test: ", len(train_pc_seqs), len(eval_pc_seqs), len(test_pc_seqs))

    train_pcs =[] 
    eval_pcs=[]
    test_pcs = []
    print ("train_pcs")
    for pc_seq in train_pc_seqs:
        train_pcs +=pc_seq
    print ("eval_pcs")
    for pc_seq in  eval_pc_seqs:
        eval_pcs +=pc_seq
    print ("test_pcs")
    for pc_seq in test_pc_seqs:
        test_pcs +=pc_seq
    
    train_pcs = np.array(train_pcs)
    eval_pcs = np.array(eval_pcs)
    test_pcs = np.array(test_pcs)
    
    np.save(train_fn, train_pcs)
    np.save(eval_fn,  eval_pcs)
    np.save(test_fn,  test_pcs)

    return train_pcs, eval_pcs, test_pcs


pc_seqs1 = get_pc_seqs_from_DFAUST("../../../data/DFAUST/registrations_f.hdf5")
pc_seqs2 = get_pc_seqs_from_DFAUST("../../../data/DFAUST/registrations_m.hdf5")

pc_seqs = pc_seqs1 + pc_seqs2

print ("Load", len(pc_seqs1), "+", len(pc_seqs2),"model sequences in total.")

train_fn ="../../data/DFAUST/train"
eval_fn ="../../data/DFAUST/eval"
test_fn ="../../data/DFAUST/test"

#mean_fn = "../../data/DFAUST/mean"
#std_fn = "../../data/DFAUST/std"



train_pcs, eval_pcs, test_pcs = split_train_and_eval_and_test_data(pc_seqs, train_fn, eval_fn, test_fn)

#pcs_mean  = train_pcs.mean(0)
#pcs_std = train_pcs.std(0)

#np.save(mean_fn, pcs_mean)
#np.save(std_fn, pcs_std)






