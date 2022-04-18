import argparse
import torch
import gym

# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/config.py
class Config:
    # DEVICE = torch.device('cpu')
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.optimizer_policy_fn = None
        self.optimizer_value_fn = None
        self.optimizer_nu_fn = None
        self.network_fn = None
        self.discount = None
        self.log_dir = ""
        self.log_level = 0
        self.tag = 'vanilla'
        self.tasks = False
        self.batch_size = 64,
        self.image_compression_dim = 200,
        self.img_dim = (480, 640),
        self.init_trajectory = "./data/",

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
