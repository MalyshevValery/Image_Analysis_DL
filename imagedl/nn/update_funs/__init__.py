"""Dict with alternative optimisation function"""
from typing import Type, Dict

from torch.optim import Optimizer

from imagedl.nn.optim.sam import SAM
from imagedl.utils.types import UpdateFunConstructor
from .sam_update import sam_update_function

update_functions: Dict[Type[Optimizer], UpdateFunConstructor] = {
    SAM: sam_update_function
}
