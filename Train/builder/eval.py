import torch
from Train.steps import finetune_eval


def build_evals(cfg):
    v = cfg.get("validation", None)

    if v:
        return finetune_eval

    else:
        return None
