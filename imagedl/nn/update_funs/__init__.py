from imagedl.nn.optim.sam import SAM
from .sam_update import sam_update_function

update_functions = {
    SAM: sam_update_function
}
