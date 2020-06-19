"""Global variables"""
import torch

DEVICE = torch.device('cuda:1')
WORKERS = 2
DISTRIBUTED = None  # ['cuda:2', 'cuda:1']

# Running Average
EPOCH_BOUND = False
ALPHA = 0.9
