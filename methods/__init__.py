from .finetune import Finetune
from .ewc import EWC
from .target import TARGET
from .ours import OURS


def get_learner(model_name, args):
    name = model_name.lower()
    if name == "ewc":
        return EWC(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "target":
        return TARGET(args)
    elif name == "ours":
        return OURS(args)
    else:
        assert 0
