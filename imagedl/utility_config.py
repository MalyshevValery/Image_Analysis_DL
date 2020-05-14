"""Global variables"""
import torch

DEVICE = torch.device('cpu')
WORKERS = 4

# Running Average
EPOCH_BOUND = False
ALPHA = 0.9
