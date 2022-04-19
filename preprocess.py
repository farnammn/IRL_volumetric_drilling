from argparse import ArgumentParser
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from bisect import bisect_left


def pose_to_matrix(pose):
    quat_norm = np.linalg.norm(pose[:, 3:], axis=-1)
    assert np.all(np.isclose(quat_norm, 1.0))
    r = R.from_quat(pose[:, 3:]).as_matrix()
    t = pose[:, :3]
    tau = np.identity(4)[None].repeat(pose.shape[0], axis=0)
    tau[:, :3, :3] = r
    tau[:, :3, -1] = t

    return tau

class DataSet:
    def __init__(self, config):
        self.image_dim = config.image_dim
        self.init_trajectory = config.init_trajectory
        self.files_list = os.listdir(config.init_trajectory)
        print(self.files_list)
        self.batch_size = config.batch_size
        self.file_index = 0
        self.data_index = 0
        self.N = len(self.files_list)
        self.data = {}
        # self.state_keys = ["l_img", "r_img", "depth", "segm", "pose_cam", "pose_drill"]
        self.image_compression_dim = config.image_compression_dim
        self.num_colors = 5
        self.color_dict = []
        self.color_idx = 0
        # self.reward_map = config.reward_map
        self.reward_list = [1, 2, 3, 4, 5]
        self.segm_color = []

    def plot(self, l_img, r_img, depth, segm):
        plt.subplot(221)
        plt.imshow(l_img)
        plt.subplot(222)
        plt.imshow(r_img)
        plt.subplot(223)
        plt.imshow(depth, vmax=1)
        plt.subplot(224)
        plt.imshow(segm)

        plt.show()

    def process_file(self, file_name):
        file = h5py.File(self.init_trajectory + file_name, 'r')

        # print(list(file['data']["l_img"][()]))
        if "data" in file.keys() and "metadata" in file.keys() and "voxels_removed" in file.keys():
            if "l_img" in file["data"].keys():
                # extrinsic = file['metadata']['camera_extrinsic'][()]
                # pose_cam = pose_to_matrix(file['data']['pose_main_camera'][()])
                # pose_cam = np.matmul(pose_cam, np.linalg.inv(extrinsic)[None])
                # pose_cam = pose_cam.reshape(pose_cam.shape[0], -1)
                print(file.keys())
                pose_drill = file['data']['pose_mastoidectomy_drill'][()]
                time = file["data"]["time"][()]
                l_img = file["data"]["l_img"][()]
                r_img = file["data"]["r_img"][()]
                depth = file["data"]["depth"][()]
                depth = depth.reshape(depth.shape + (1,))
                segm = file["data"]["segm"][()]
                for seg in segm:
                    try:
                        idx = [np.array_equal(segm, x) for x in self.segm_color].index(True)
                    except:
                        self.segm_color.append(seg)
                        print(self.segm_color)
                        with open("colors.txt", "a") as f:
                            f.write(segm)
                            f.close()
                # imgs = np.concatenate((l_img, r_img, depth, segm), axis=-1)
                # pca, imgs_rep = self.PCA(imgs)
                # print("imgs_rep_shape: ", imgs_rep.shape)
                # state_rep = np.concatenate((imgs_rep, pose_drill), axis=-1)

                state_rep = []
                if "voxel_removed" in file["voxels_removed"]:
                    voxel_time_stamp = file["voxels_removed"]["time_stamp"][()]
                    # voxel_removed = file["voxels_removed"]["voxel_removed"][()]
                    voxel_color = file["voxels_removed"]["voxel_color"][()]

                else:
                    # voxel_removed = []
                    voxel_color = []
                    voxel_time_stamp = []
                rewards, cum_rewards = self.color_pres(voxel_color, voxel_time_stamp, time)
                #state_rep = np.concatenate((imgs_rep, cum_rewards), axis=-1)
                # cam_change = np.concatenate((pose_cam[0].reshape(pose_cam[0].shape + (1,)), pose_cam[1:] - pose_cam[:-1]), axis=0)
                drill_change = np.concatenate((pose_drill[0].reshape((1,)+ pose_drill[0].shape), pose_drill[1:] - pose_drill[:-1]), axis=0)
                # action_rep = np.concatenate((cam_change.reshape(cam_change.shape[0], -1),
                #                                 drill_change.reshape(cam_change.shape[0], -1)), axis=-1)
                data = {"time": time, "state_rep": state_rep, "action_rep": drill_change, "reward": rewards, "l_img": l_img, "r_img": r_img}
                return data
            else:
                return {}

    def process_data(self):
        data = []
        rewards = []
        for file in self.files_list:
            data_file = self.process_file(file)
            if len(data_file.keys()) != 0:
                rewards.append(np.sum(data_file["reward"]))
                data.append(data_file)

        data = [d for _, d in sorted(zip(rewards, data), key=lambda pair: pair[0])]
        return data, sorted(rewards)

    def PCA(self, image_array):
        image_array = image_array.reshape(image_array.shape[0], -1)
        pca = PCA(self.image_compression_dim)
        image_transformed = pca.fit_transform(image_array)
        return pca, image_transformed

    def color_pres(self, voxel_color, time_stamps, times):
        # freq = np.zeros(self.num_colors)
        # color_pres = np.zeros(len(times), self.num_colors)
        rewards = np.zeros(len(times))
        reward = 0
        cum_rewards = np.zeros(len(times))
        for time_stamp, color in zip(time_stamps, voxel_color):

            try:
                idx = [np.array_equal(color,x) for x in self.color_dict].index(True)
                reward = self.reward_list[idx]
            except:
                self.color_dict.append(color)
                reward = self.reward_list[self.color_idx]
                self.color_idx += 1
                # freq[self.color_idx] += 1

            idx = bisect_left(times, time_stamp)
            if idx:
                rewards[idx] = reward
                if idx != 0:
                    cum_rewards[idx] = cum_rewards[idx - 1] + reward
                else:
                    cum_rewards[idx] = reward

        return rewards, cum_rewards

