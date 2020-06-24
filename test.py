import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from imagedl.utility_config import DISTRIBUTED


def setup(world_size, rank):
    # initialize the process group
    dist.init_process_group("nccl", init_method='file:///tmp/nccl.dat',
                            world_size=world_size, rank=rank)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, distributed):
    print(f"Running basic DDP example on rank {rank}.")
    world_size = len(distributed)
    device = distributed[rank]
    torch.cuda.set_device(device)
    setup(world_size, rank)

    # create model and move it to GPU with id rank
    model = ToyModel().to(device)
    print(device)
    ddp_model = DDP(model, device_ids=[device])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    print(outputs.shape, outputs.device)

    cleanup()


def run_demo(demo_fn):
    mp.spawn(demo_fn,
             args=(DISTRIBUTED,),
             nprocs=len(DISTRIBUTED),
             join=True)


if __name__ == "__main__":
    run_demo(demo_basic)
