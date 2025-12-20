import torch
from Train.steps import mae_step, simclr_step, finetune_step

def build_steps(cfg):
    