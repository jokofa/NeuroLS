#
import os
import sys
from warnings import warn
from typing import Dict, Optional, List
from timeit import default_timer as timer

import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "get_lambda_decay",
    "get_activation_fn",
    "get_norm",
    "timeit",
    "rm_from_kwargs",
    "recursive_str_lookup",
    "count_parameters",
    "format_ds_save_path"
]


def get_lambda_decay(schedule_type: str, decay: float, decay_step: Optional[int] = None):
    """Create learning rate scheduler (different strategies)."""

    if schedule_type in ['exponential', 'smooth']:
        assert 1.0 >= decay >= 0.9, \
            f"A decay factor >1 or <0.9 is not useful for {schedule_type} schedule!"

    if schedule_type == 'exponential':
        def decay_(eps):
            """exponential learning rate decay"""
            return (decay ** eps) ** eps
    elif schedule_type == 'linear':
        def decay_(eps):
            """linear learning rate decay"""
            return decay ** eps
    elif schedule_type == 'smooth':
        def decay_(eps):
            """smooth learning rate decay"""
            return (1 / (1 + (1 - decay) * eps)) ** eps
    elif schedule_type == 'step':
        assert decay_step is not None, "need to specify decay step for step decay."
        def decay_(eps):
            """step learning rate decay"""
            return decay ** (eps // decay_step)
    else:
        warn(f"WARNING: No valid lr schedule specified. Running without schedule.")
        def decay_(eps):
            """constant (no decay)"""
            return 1.0
    return decay_


def get_activation_fn(activation: str, module: bool = False, negative_slope: float = 0.01, **kwargs):
    if activation is None:
        return None
    if activation.upper() == "RELU":
        return F.relu if not module else nn.ReLU(**kwargs)
    elif activation.upper() == "GELU":
        return F.gelu if not module else nn.GELU()
    elif activation.upper() == "TANH":
        return F.tanh if not module else nn.Tanh()
    elif activation.upper() == "LEAKYRELU":
        return F.leaky_relu if not module else nn.LeakyReLU(negative_slope, **kwargs)
    else:
        raise ModuleNotFoundError(activation)


def get_norm(norm_type: str, hdim: int, **kwargs):
    if norm_type is None:
        return None
    if norm_type.lower() in ['bn', 'batch_norm']:
        return nn.BatchNorm1d(hdim, **kwargs)
    elif norm_type.lower() in ['ln', 'layer_norm']:
        return nn.LayerNorm(hdim, **kwargs)
    else:
        raise ModuleNotFoundError(norm_type)


def timeit(f):
    """
    timeit decorator to measure execution time.
    (does NOT disable garbage collection!)
    """
    def w(*args, **kwargs):
        start = timer()
        out = f(*args, **kwargs)
        t = timer() - start
        return out, t
    return w


def rm_from_kwargs(kwargs: Dict, keys: List):
    """Remove specified items from kwargs."""
    keys_ = list(kwargs.keys())
    for k in keys:
        if k in keys_:
            del kwargs[k]
    return kwargs


def recursive_str_lookup(d: Dict):
    """Return all str values of a possibly nested dict."""
    for value in d.values():
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict) and value:
            yield from recursive_str_lookup(value)


def count_parameters(model: nn.Module, trainable: bool = True):
    """Count the number of (trainable) parameters of the provided model."""
    if trainable:
        model.train()   # set to train mode
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())



def format_ds_save_path(directory, args=None, affix=None, fname=''):
    """Format the path for saving datasets"""
    directory = os.path.normpath(os.path.expanduser(directory))

    if args is not None:
        for k, v in args.items():
            if isinstance(v, str):
                fname += f'_{v}'
            else:
                fname += f'_{k}_{v}'

    if affix is not None:
        fname = str(affix) + fname
    if fname != '':
        fpath = os.path.join(directory, fname)
    else:
        fpath = directory
    if fpath[-3:] not in ['.pt', 'dat', 'pkl']:
        fpath += '.pt'

    if os.path.isfile(fpath):
        print('Dataset file with same name exists already. Overwrite file? (y/n)')
        a = input()
        if a != 'y':
            print('Could not write to file. Terminating program...')
            sys.exit()

    return fpath
