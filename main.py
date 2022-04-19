from preprocess import DataSet
from config import Config
from irl import irl
from utils import *
from os import mkdir
import torch


file_list = [
    "20220407_170930.hdf5",
    "20220407_171452.hdf5",
    "20220407_171709.hdf5",
    "20220407_171806.hdf5",
    "20220407_171928.hdf5"
]

kwargs = {
    "batch_size": 64,
    "image_compression_dim": 200,
    "image_dim": (480, 640),
    "init_trajectory": "./data/",
    "mu_bins": len(file_list),
    "state_dim": 208,
    "action_dim": 16,
    "lr": 0.01,
    "num_steps": 30000,
    "log_dir": "./exp/" + get_time_str() + "/",
    "is_rew_fix": False
}
mkdirs(kwargs["log_dir"])

kwargs.setdefault('log_level', 0)
config = Config()
config.merge(kwargs)

data_set = DataSet(files_list=file_list, config=config)
data_list, VaRs = data_set.process_data()
irl(data_list=data_list, Rs=VaRs, config=config)
