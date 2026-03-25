import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from datasets.CD_dataset import ImageDataset 

# dataset = ImageDataset(
#     root_dir=r'C:\Users\Alex.zeng\Desktop\BIT_CD\LEVIR_CD',  # 比如 'C:/data/LEVIR'
#     split='test',
#     img_size=256,
#     is_train=False,
#     to_tensor=True   # 先设为 False，这样增强后还是 numpy，容易打印
# )

# for i in range(3):
#     sample = dataset[i]
#     print('-'*50)

# checkpoint = torch.load(r"C:\Users\Alex.zeng\Desktop\BIT_CD\checkpoints\CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_trainval_test_200_linear\best_ckpt.pt",weights_only=False)
# print(checkpoint.keys())
