import argparse
import torch
import gym

# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/config.py
class Config:
    DEVICE = torch.device('gpu')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_policy_fn = None
        self.optimizer_value_fn = None
        self.optimizer_nu_fn = None
        self.network_fn = None
        self.discount = None
        self.log_dir = ""
        self.log_level = 0
        self.tag = 'vanilla'
        self.gradient_clip = None
        self.entropy_weight = 0
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.__eval_env = None
        self.eval_interval = 0
        self.eval_episodes = None
        self.tasks = False
        self.is_spectral = False
        self.is_a2c = False
        self.dist = None
        self.alpha = 0
        self.mean = 0
        self.std_dev = 0
        self.policy_lr = 3e-2
        self.nu_lr = 3e-2
        self.value_lr = 5e-4
        self.Vmin = 0
        self.Vmax = 1000
        self.N = 10
        self.alpha_bins = 15
        self.env_width = 7
        self.env_height = 7
        self.r_goal = 12
        self.r_cliff = -5
        self.r_remaining = -1
        self.episode_length = 10
        self.value_times = 3

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
