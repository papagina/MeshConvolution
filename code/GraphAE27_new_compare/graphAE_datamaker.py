import graphAE_dataloader as loader
import os
import numpy as np





def split_train_and_eval_and_test_data(src_folder, train_fn, eval_fn, test_fn ):
    pcs = loader.get_pcs_from_ply_folder(src_folder)
    pcs = pcs/1000.0
    
    np.random.shuffle(pcs)
    num = pcs.shape[0]
    train_pcss =pcs[0:int(num*0.8)]
    eval_pcss = pcs[int(num*0.8):int(num*0.9)]
    test_pcss = pcs[int(num*0.9):]
    
    
    np.save(train_fn, train_pcss)
    np.save(eval_fn,  eval_pcss)
    np.save(test_fn,  test_pcss)
                
            
#process_face_data()
    
src_folder="../../data/data1/plys/"
train_fn ="../../data/data1/train"
eval_fn ="../../data/data1/eval"
test_fn ="../../data/data1/test"

split_train_and_eval_and_test_data(src_folder, train_fn, eval_fn, test_fn )
#pc = loader.get_pc_from_ply_fn("/diskb/body/data/COMA/bareteeth.000001.ply")   
#np.save("/diskb/body/data/COMA/bareteeth.000001", pc)
    
    

































