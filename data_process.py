from abc import ABC, abstractmethod
from grid_world import GridWorld
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import os
import numpy as np
from sklearn.cluster import KMeans
import copy
import pickle


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
            data = {"state_rep": [], "action_rep": [], "reward": []}
            state = self.reset()

            while (not done):
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
    def __init__(self, config):


        self.data_path = Path(__file__).resolve().parent

        # data_processed = load_data(data_path, "human_robot_data_p1_train.mat")
        self.l = config.driving_l
        # x_follower = data_processed["x_f"]
        # vx_follower = data_processed["vx_f"]
        # ax_follower = data_processed["ax_f"]
        # x_leader = data_processed["x_l"]
        # ay_follower = data_processed["ay_f"]
        # vy_follower = data_processed["vy_f"]
        # y_leader = data_processed["x_f"]
        # y_follower = data_processed["y_f"]
        # dt = data_processed["dt"]
        #
        # a, b, c = self.creat_dataset(x_follower, vx_follower, ax_follower, x_leader, ay_follower, vy_follower, y_leader,
        #                              y_follower, 0.0167, l)
        #
        # self.data_list = self.make_trajectory(a, b, c, w_s=1)

    def return_data_list(self):
        d_list = []
        raw_list = self.list_path(self.data_path)
        for data_processed in raw_list:
            x_follower = data_processed["x_f"]
            vx_follower = data_processed["vx_f"]
            ax_follower = data_processed["ax_f"]
            x_leader = data_processed["x_l"]
            ay_follower = data_processed["ay_f"]
            vy_follower = data_processed["vy_f"]
            y_leader = data_processed["x_f"]
            y_follower = data_processed["y_f"]
            dt = data_processed["dt"]
            a, b, c = self.creat_dataset(x_follower, vx_follower, ax_follower, x_leader, ay_follower, vy_follower,
                                         y_leader, y_follower, 0.0167, self.l)
            d_list.append(self.make_trajectory(a, b, c, w_s=1))
        return d_list

    def list_path(self, main_path):
        data_process_list = []
        listpath = []
        for i in range(10):
            listpath.append("human_robot_data_p" + str(i+1) + "_train.mat")
        for file_path in listpath:
            data_process_list.append(loadmat(os.path.join(main_path, "data/" + file_path)))
        return data_process_list






    def create_state(self, x_f, vx_f, ax_f, x_l, ay_f, vy_f, y_l, y_f, dt, l):
        """
        calculates other states then, separate state and action
        return: two numpy arrays
        1 - state: the state contains x_f, y_f, theta_f, x_l, y_l respectively
        The dimention is 5 times lenght of the whole trajectory.
        To do : add three more dimension regard to leader
        2- actions : u_a, u_s. The dimension is 2 times lenght of the whole trajectory

        """
        x_l = x_l + 140.
        y_l = -(y_l - 3.)
        x_f = x_f + 140;
        y_f = -(y_f - 3.)
        vy_f = -vy_f
        ay_f = -ay_f
        nt = x_f.shape[1]
        theta_f = np.arctan(vy_f / vx_f)
        v_f = np.sqrt(np.power(vx_f, 2) + np.power(vy_f, 2))
        u_a = ax_f * np.cos(theta_f) + ay_f * np.sin(theta_f)
        theta_fd = (ay_f * np.cos(theta_f) - ax_f * np.sin(theta_f)) / v_f
        thetafdd = (theta_fd[:, 1:] - theta_fd[:, :nt - 1])
        axu = thetafdd[0, -1].reshape(1, 1)
        thetafdd = np.concatenate((thetafdd, axu), axis=1) / dt
        delta_f = np.arctan(-l * theta_fd / v_f)

        delta_fd = (-l / np.power(v_f, 2)) * (thetafdd * v_f - theta_fd * u_a) * (
                np.power(v_f, 2) / (np.power(v_f, 2) + np.power(l * theta_fd, 2)))
        u_s = delta_fd
        return np.concatenate([x_f, y_f, theta_f, x_l, y_l], axis=0), np.concatenate([u_a, u_s], axis=0)

    def descretize(self, action_space, n_c=2):
        """desceritize the action space by kmeans clustering
        in other words, it maps each action to its center.
        input: n_c --> number of clusters
        action space: should be in shape n_samples * 2
        output: descretized action_space

        """
        aux = copy.deepcopy(action_space)
        kmeans = KMeans(n_clusters=n_c, random_state=0).fit(aux.reshape(-1, 2))
        centers = kmeans.cluster_centers_

        lab = kmeans.labels_
        for i in range(lab.shape[0]):
            k = lab[i]
            action_space[0, i] = centers[k, 0]
            action_space[1, i] = centers[k, 1]
        return action_space

    def reward(self, x_f, y_f, theta_f, x_l, y_l, u_a, r_1=0.05, r_2=1., r_3=0.1, r_4=1.0, r_5=0.1, r_6=0.5):
        """
        calculate cost fucntion
        """
        x_rel = x_l - x_f
        y_rel = y_l - y_f
        #     v_rel =
        f1 = (x_rel > 2.5) * (np.log(1 + np.exp(r_1 * (x_rel - 2.5))) - np.log(2))  # not too far from leader
        f2 = (x_rel < 2.5) * (np.log(1 + np.exp(-r_2 * (x_rel - 2.5))) - np.log(2))  # leader should remain first
        #     f3 =
        #     f4 =
        f5 = np.log(1 + np.exp(r_5 * np.abs(y_rel))) - np.log(2)
        f6 = (y_f > 2) * (np.log(1 + np.exp(r_6 * (y_f - 2.))) - np.log(2)) + (y_f < -2) * (
                np.log(1 + np.exp(-r_6 * (y_f + 2.))) - np.log(2))
        f = f1 + f2 + f5 + f6
        return - f.reshape(1, -1)

    def creat_dataset(self, x_f, vx_f, ax_f, x_l, ay_f, vy_f, y_l, y_f, dt, l):
        state, action = self.create_state(x_f, vx_f, ax_f, x_l, ay_f, vy_f, y_l, y_f, dt, l)
        action = self.descretize(action, n_c=15)
        cost_p = self.reward(state[0, :], state[1, :], state[2, :], state[1, :], state[1, :], action[0, :], r_1=0.05,
                             r_2=1.,
                             r_3=0.1, r_4=1.0, r_5=0.1, r_6=0.5)
        return state, action, cost_p

    # Building trajectory
    # chunk the wole data to different trajectories
    def make_trajectory(self, state_sp, action_sp, cost_p, w_s=30):
        """ Chunk the whole trajectory to the trajectories with size w_s"""
        nn = state_sp.shape[1]
        n_t = nn // w_s

        for i in range(n_t):
            tr_s = state_sp[:, w_s * i: w_s * (i + 1)]
            tr_s = np.expand_dims(tr_s, axis=0)
            tr_a = action_sp[:, w_s * i: w_s * (i + 1)]
            tr_a = np.expand_dims(tr_a, axis=0)
            tr_c = cost_p[:, w_s * i: w_s * (i + 1)]
            tr_c = np.expand_dims(tr_c, axis=0)
            if i == 0:
                traj_s = tr_s
                traj_a = tr_a
                traj_c = tr_c
            else:
                traj_s = np.concatenate((traj_s, tr_s), axis=0)
                traj_a = np.concatenate((traj_a, tr_a), axis=0)
                traj_c = np.concatenate((traj_c, tr_c), axis=0)

        # reshape state, action and reward n_traj, lentgh, n_state
        st_sh = traj_s.shape
        ac_sh = traj_a.shape
        re_sh = traj_c.shape
        traj_s = traj_s.reshape((-1, st_sh[2], st_sh[1]))
        traj_a = traj_a.reshape((-1, ac_sh[2], ac_sh[1]))
        traj_c = traj_c.reshape((-1, re_sh[2], re_sh[1]))
        dict_tra = {}
        dict_tra["state_rep"] = traj_s[:, 0, :]
        dict_tra["action_rep"] = traj_a[:, 0, :]
        dict_tra["reward"] = np.average(traj_c, axis=-1)
        # dict_tra["reward"] = traj_c[:, 0, :]

        return dict_tra
