r"""This module provides package-wide configuration management."""
from typing import Any, List
import time
import os, sys
from yacs.config import CfgNode as CN

class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):
        self._C = CN()
        self._C.RANDOM_SEED = 0
        self._C.PHASE = "training"
        self._C.EXPERIMENT_NAME = "default"
        self._C.RESULTS_DIR = "results"
        self._C.OVERFIT= False

        self._C.SHAPENET_DATA = CN()
        self._C.SHAPENET_DATA.PATH = '/scratch/jiadeng_root/jiadeng/shared_data/datasets/ShapeNetCore.v1/'
        # self._C.SHAPENET_DATA.TRANSFORM = None

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 4
        self._C.OPTIM.VAL_BATCH_SIZE = 16
        self._C.OPTIM.WORKERS = 4
        self._C.OPTIM.EPOCH = 2
        self._C.OPTIM.LR = 0.015
        self._C.OPTIM.MOMENTUM = 0.9
        self._C.OPTIM.WEIGHT_DECAY = 0.001
        self._C.OPTIM.CLIP_GRADIENTS = 12.5

        self._C.GCC = CN()
        self._C.GCC.INPUT_MESH_FEATS = 3
        self._C.GCC.HIDDEN_DIMS = [32, 64, 128]
        self._C.GCC.CLASSES = 57
        self._C.GCC.CONV_INIT = "normal"

        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)
        
        self._C.CKP = CN()
        self._C.CKP.full_experiment_name = ("exp_%s_%s" % ( time.strftime("%m_%d_%H_%M_%S"), self._C.EXPERIMENT_NAME) )
        self._C.CKP.experiment_path = os.path.join(self._C.RESULTS_DIR, self._C.CKP.full_experiment_name)
        self._C.CKP.best_loss = sys.float_info.max
        self._C.CKP.best_acc = 0.

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        common_string: str = str(CN({"RANDOM_SEED": self._C.RANDOM_SEED})) + "\n"
        common_string += str(CN({"DATA": self._C.SHAPENET_DATA})) + "\n"
        common_string += str(CN({"BASE_MODEL": self._C.GCC})) + "\n"
        common_string += str(CN({"OPTIM": self._C.OPTIM})) + "\n"
        common_string += str(CN({"CHECKPOINT": self._C.CKP})) + "\n"
        return common_string

    def __repr__(self):
        return self._C.__repr__()

