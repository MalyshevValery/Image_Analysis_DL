"""Global variables"""
import torch

DEVICE = torch.device('cuda:0')
WORKERS = 4
DISTRIBUTED = None  # ['cuda:2', 'cuda:1']
TB_ITER = 10

# Running Average
EPOCH_BOUND = True
ALPHA = 0.9
