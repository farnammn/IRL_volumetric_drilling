from argparse import ArgumentParser
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from bisect import bisect_left
from utils import get_time_str


def pose_to_matrix(pose):
    quat_norm = np.linalg.norm(pose[:, 3:], axis=-1)
    assert np.all(np.isclose(quat_norm, 1.0))
    r = R.from_quat(pose[:, 3:]).as_matrix()
    t = pose[:, :3]
    tau = np.identity(4)[None].repeat(pose.shape[0], axis=0)
    tau[:, :3, :3] = r
    tau[:, :3, -1] = t

    return tau

def view_data(l_img, r_img, name):
    plt.subplot(221)
    plt.imshow(l_img)
    plt.subplot(222)
    plt.imshow(r_img)
    # plt.subplot(223)
    # plt.imshow(depth[i], vmax=1)
    # plt.subplot(224)
    # plt.imshow(segm[i])

    plt.savefig("img_"+str(name)+ "_" + get_time_str()+".png")

class DataSet:
    def __init__(self, config):
        self.image_dim = config.image_dim
        self.init_trajectory = config.init_trajectory
        self.files_list = os.listdir(config.init_trajectory)
        self.batch_size = config.batch_size
        self.data_list = []
        self.time = 1e15
        # self.state_keys = ["l_img", "r_img", "depth", "segm", "pose_cam", "pose_drill"]
        self.image_compression_dim = config.image_compression_dim
        self.color_dict = []
        self.color_idx = 0
        self.reward_map = config.reward_map
        self.sensitive_reward = config.sensitive_reward

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
        try:
            file = h5py.File(self.init_trajectory + file_name, 'r')
        except:
            print("File reading error")
            return


        if "data" in file.keys() and "metadata" in file.keys() and "voxels_removed" in file.keys():
            if "l_img" in file["data"].keys() and "pose_mastoidectomy_drill" in file["data"].keys():
                # extrinsic = file['metadata']['camera_extrinsic'][()]
                # pose_cam = pose_to_matrix(file['data']['pose_main_camera'][()])
                # pose_cam = np.matmul(pose_cam, np.linalg.inv(extrinsic)[None])
                # pose_cam = pose_cam.reshape(pose_cam.shape[0], -1)
                pose_drill = file['data']['pose_mastoidectomy_drill'][()]
                time = file["data"]["time"][()]
                l_img = file["data"]["l_img"][()]
                r_img = file["data"]["r_img"][()]
                depth = file["data"]["depth"][()]
                depth = depth.reshape(depth.shape + (1,))
                segm = file["data"]["segm"][()]
                # imgs = np.concatenate((l_img, r_img, depth, segm), axis=-1)
                # pca, imgs_rep = self.PCA(imgs)
                # print("imgs_rep_shape: ", imgs_rep.shape)
                # state_rep = np.concatenate((imgs_rep, pose_drill), axis=-1)

                state_rep = []
                if "time_stamp" in file["voxels_removed"] and "voxel_color" in file["voxels_removed"]:
                    voxel_time_stamp = file["voxels_removed"]["time_stamp"][()]
                    # voxel_removed = file["voxels_removed"]["voxel_removed"][()]
                    voxel_color = file["voxels_removed"]["voxel_color"][()]
                else:
                    voxel_color = []
                    voxel_time_stamp = []
                rewards = self.color_pres(voxel_color, voxel_time_stamp, time)
                # cam_change = np.concatenate((pose_cam[0].reshape(pose_cam[0].shape + (1,)), pose_cam[1:] - pose_cam[:-1]), axis=0)
                drill_change = np.concatenate((pose_drill[0].reshape((1,)+ pose_drill[0].shape), pose_drill[1:] - pose_drill[:-1]), axis=0)
                # action_rep = np.concatenate((cam_change.reshape(cam_change.shape[0], -1),
                #                                 drill_change.reshape(cam_change.shape[0], -1)), axis=-1)
                data = {"time": time, "state_rep": state_rep, "action_rep": drill_change, "reward": rewards, "l_img": l_img, "r_img": r_img}
                if time[0] < self.time:
                    self.data_list.append(data)
                else:
                    print(len(self.data_list))
                    data_2 = self.data_list[-1]
                    for key in data_2:
                        data_2[key] = np.concatenate((data_2[key], data[key]), axis=0)
                    self.data_list[-1] = data_2
                self.time = time[-1]

    def process_data(self):
        for file in self.files_list:
            self.process_file(file)
        return self.data_list

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
            flag = False
            for color2, r in self.reward_map:
                if np.array_equal(color2, color):
                    reward = r
                    flag = True
                    break
            if not flag:
                reward = self.sensitive_reward
                print(color)
            idx = bisect_left(times, time_stamp)
            if idx and idx < len(times):
                rewards[idx] = reward
        return rewards

