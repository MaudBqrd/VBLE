from .models import *

cae_models = {
    "mbt": JointAutoregressiveHierarchicalPriors,
    "cheng": Cheng2020Attention,
}

cae_configs = {
    "mbt": {"N": 192, "M": 320},
    "cheng": {"N": 192},
}