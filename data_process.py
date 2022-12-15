from abc import ABC, abstractmethod
from grid_world import GridWorld
import numpy as np
class Dataset(ABC):
    @abstractmethod
    def return_data_list(self):
        pass




class GymSet(Dataset):
    def __init__(self, config):
        if config.env_id == "gridWorld":
            self.env = GridWorld(config.env_width, config.env_length)

        self.num_traj = config.num_traj
        self.action_dim = config.action_dim
        self.state_dim = config.env_width * config.env_length
        self.data_list = []
        self.expert_policy = self.compute_expert_policy()
        self.run_env()

    def run_env(self):
        for _ in range(self.num_traj):
            done = False
            data = {"state_rep":[], "action_rep":[], "reward":[]}
            state = self.reset()

            while(not done):
                action = self.expert_policy[state]
                data["state_rep"].append(self.index_to_one_hat(state, self.state_dim))
                data["action_rep"].append(self.index_to_one_hat(action, self.action_dim))
                state, reward, done, _ = self.env.step(action)
                data["reward"].append(reward)
            self.data_list.append(data)

    def index_to_one_hat(self, action, size):
        ret = np.zeros(size)
        ret[action] = 1
        return ret

    def compute_expert_policy(self):
        return np.random.randint(self.action_dim, size=self.state_dim)


    def reset(self):
        return self.env.reset()

    def return_data_list(self):
        return self.data_list


class SimDrivingSet(Dataset):
    def return_data_list(self):
        pass