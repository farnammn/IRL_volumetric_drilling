import gym
import numpy as np
from gym import spaces
import random
import copy

import matplotlib.pyplot as plt


directions = {0: ["up", [-1, 0]],
              1: ["down", [1, 0]],
              2: ["left", [0, -1]],
              3: ["right", [0, 1]]}
p_chosen = 0.75  # probability of moving in chosen direction


class GridWorld(gym.Env):
    """Cliff walking Environment that follows gym interface
    Parameters from:
    https://arxiv.org/ftp/arxiv/papers/1203/1203.3497.pdf
    Sutton and Barto (pg 158):
    https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, width=8, length=8,
                 r_remaining=-1):
        super(GridWorld, self).__init__()

        self.shape = [width, length]
        print("Size of grid: ", self.shape)

        start = (self.shape[0] - 1, 0)  # bottom left, right
        self.start_index = np.ravel_multi_index(start, self.shape)
        self.current_state_index = self.start_index  #  initial position

        # Cliff Location
        self.map = np.zeros(self.shape, dtype=np.int)
        self.map[self.shape[0] - 1, 1:-1] = 1
        self.map[self.shape[0] - 1, self.shape[1] - 1] = 2

        self.rewards = [-1, -5, 12]
        self.terminals = [False, True, True]

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.MultiDiscrete(self.shape)

    def check_in_grid(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord: list: [x, y]
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def step(self, action):
        """
        :param action: int: 0/1/2/3 for up/down/left/right resp.
        """
        current_state = np.unravel_index(self.current_state_index, self.shape)

        def get_action(chosen_action):
            """
            :param chosen_action: int: 0/1/2/3 for the chosen action
            :return:
            """
            if not chosen_action in directions.keys():
                raise ValueError("Invalid action: {}.".format(chosen_action))
            p = random.random()
            if p < p_chosen:
                return directions[chosen_action]
            else:
                remaining_actions = copy.deepcopy(directions)
                remaining_actions.pop(chosen_action)
                return random.choices(list(remaining_actions.values()))[
                    0]  # choose one of the other actions with equal probability

        action = get_action(chosen_action=action)
        # print("Action taken: ", action)
        delta = action[1]
        new_position = np.array(current_state) + np.array(delta)
        new_position = self.check_in_grid(new_position).astype(int)
        new_state_index = np.ravel_multi_index(tuple(new_position), self.shape)

        cell_type = self.map[tuple(new_position)]
        is_done = self.terminals[cell_type]
        if is_done:
            self.current_state_index = self.start_index
        else:
            self.current_state_index = new_state_index
        return self.index_to_one_hat(self.current_state_index), self.rewards[cell_type], is_done, {}


    def index_to_one_hat(self, index):
        out = np.zeros(np.prod(self.shape))
        out[index] = 1
        return out

    def reset(self):
        self.current_state_index = self.start_index
        return self.index_to_one_hat(self.current_state_index)

