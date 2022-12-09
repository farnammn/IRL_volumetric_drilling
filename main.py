from volumetric_drilling.preprocess import DataSet
from config import Config
from irl import irl
from utils import *

kwargs = {
    "batch_size": 64,
    "image_compression_dim": 200,
    "image_dim": (480, 640),
    "init_trajectory": "./data/",
    "state_dim": 200 + 7 + 1,
    "action_dim": 16,
    "lr": 0.01,
    "num_steps": 30000,
    "log_dir": "./exp/" + get_time_str() + "/",
    "is_rew_fix": False,
    "reward_map": [
        ([255, 249, 219, 255], -0.5), #cream #normal_bone
        ([110, 184, 209, 255], -10), #lightblue #back_area
        ([255, 225, 214, 255], -10), #cream #sensitive area
        ([100,   0,   0, 255], -20), #red sensitive area
        ([0, 255, 149, 255], +100), #lightgrean target
        ([233, 0, 255, 255], +100), #purple target
    ],
    "sensitive_reward": -10,
}
mkdirs(kwargs["log_dir"])

kwargs.setdefault('log_level', 0)
config = Config()
config.merge(kwargs)

data_set = DataSet(config=config)
data_list = data_set.return_data_list()

####
# list of trajectories
# data_list = [data] * number of trajectories
# data is each trajectory
# data = {"state_rep" : , "action_rep": , "reward" : }
####

irl(data_list=data_list, config=config)

