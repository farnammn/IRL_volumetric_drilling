from preprocess import DataSet
from config import Config
from irl import irl
from utils import *
from os import mkdir
import torch

kwargs = {
    "batch_size": 64,
    "image_compression_dim": 200,
    "image_dim": (480, 640),
    "init_trajectory": "./data/",
    "state_dim": 208,
    "action_dim": 16,
    "lr": 0.01,
    "num_steps": 30000,
    "log_dir": "./exp/" + get_time_str() + "/",
    "is_rew_fix": False,
}
mkdirs(kwargs["log_dir"])

kwargs.setdefault('log_level', 0)
config = Config()
config.merge(kwargs)

data_set = DataSet(config=config)
data_list, VaRs = data_set.process_data()
# irl(data_list=data_list, Rs=VaRs, config=config)
