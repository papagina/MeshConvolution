import graphAE_dataloader as loader
import os
import numpy as np


ply_fn = "/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/body/fall/data/data_test_mem_cost/sample_440k.ply"
np_fn = "/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/body/fall/data/data_test_mem_cost/440k_"

pc = loader.get_pc_from_ply_fn(ply_fn)

pcs = np.array([pc, pc, pc, pc])

np.save(np_fn+"train", pcs)
np.save(np_fn+"test", pcs)
np.save(np_fn+"eval", pcs)

np.save(np_fn+"mean", pc)

































