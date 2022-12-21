from config import Config
from irl import irl, irl_dr
from utils import *
from data_process import GymSet
from data_process import SimDrivingSet

kwargs = {
    "batch_size": 64,
    "image_compression_dim": 200,
    "image_dim": (480, 640),
    "init_trajectory": "./data/",
    "state_dim": 5,
    "action_dim": 2,
    "lr": 0.01,
    "num_steps": 30000,

    "is_rew_fix": False,
    "sensitive_reward": -10,
    "num_traj": 100,
    "env_id": "gridWorld",
    "env_width": 8,
    "env_length": 8,
    "driving_l": 3.47,
    "log_dir": "./exp/" + get_time_str() + "/",
}

mkdirs(kwargs["log_dir"])

kwargs.setdefault('log_level', 0)
config = Config()
config.merge(kwargs)

# data_set = GymSet(config)
data_set = SimDrivingSet(config)


data_list = data_set.return_data_list()




####
# list of trajectories
# data_list = [data] * number of trajectories
# data is each trajectory
# data = {"state_rep" : , "action_rep": , "reward" : }
####

# irl(data_list=data_list, config=config)
irl_dr(data_list=data_list, config=config)

