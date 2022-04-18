from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


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
    def __init__(self, files_list, image_compression_dim, images_dim=(480, 640, 3), init_trajectory="./data/", batch_size=64):
        random.shuffle(files_list)
        self.files_list = files_list
        self.init_trajectory = init_trajectory
        self.batch_size = batch_size
        self.file_index = 0
        self.data_index = 0
        self.N = len(self.files_list)
        self.data_states = {}
        self.data_actions = {}
        self.state_keys = ["l_img", "r_img", "depth", "segm", "pose_cam", "pose_drill"]
        self.image_compression_dim = image_compression_dim


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


    def preprocess(self, file_name):
        file = h5py.File(self.init_trajectory + file_name, 'r')

        # print(list(file['data']["l_img"][()]))
        if "data" in file.keys() and "metadata" in file.keys() and "voxels_removed" in file.keys():
            if "l_img" in file["data"].keys():
                extrinsic = file['metadata']['camera_extrinsic'][()]
                pose_cam = pose_to_matrix(file['data']['pose_main_camera'][()])
                pose_cam = np.matmul(pose_cam, np.linalg.inv(extrinsic)[None])
                time = file["data"]["time"][()]
                l_img = file["data"]["l_img"][()]
                print(self.pca(l_img)[1])
                r_img = file["data"]["r_img"][()]
                depth = file["data"]["depth"][()]
                segm = file["data"]["segm"][()]
                pose_drill = pose_to_matrix(file['data']['pose_mastoidectomy_drill'][()])
                print(l_img[0].shape)
                print(depth[0].shape)
                print(segm.shape)

                if "voxel_removed" in file["voxels_removed"]:
                    voxel_time_stamp = file["voxels_removed"]["time_stamp"][()]
                    print(voxel_time_stamp.shape)
                    voxel_removed = file["voxels_removed"]["voxel_removed"][()]
                    voxel_color = file["voxels_removed"]["voxel_color"][()]
                else:
                    voxel_removed = []
                    voxel_color = []
                    voxel_time_stamp = []

                data_states = {"l_img": l_img, "r_img": r_img, "depth": depth, "segm": segm, "pose_cam": pose_cam,
                               "pose_drill": pose_drill, "voxel_removed": voxel_removed, "voxel_color": voxel_color,
                               "time": time, "voxel_time_stamp": voxel_time_stamp}

                cam_change = pose_cam[1:] - pose_cam[:-1]
                drill_change = pose_drill[1:] - pose_drill[:-1]
                data_actions = {"cam_change": cam_change, "drill_change": drill_change}

                return data_actions, data_states
            else:
                return {}, {}

    def next_batch(self):
        if self.data_index == 0:
            if self.file_index >= len(self.files_list) - 1:
                return False
            self.file_index += 1
            self.data_actions, self.data_states = self.preprocess(self.files_list[self.file_index])
            if len(self.data_actions.keys()) == 0:
                return self.next_batch()

        batch_actions = {}
        batch_states = {}
        for key, value in self.data_actions.items():
            batch_actions[key] = value[self.data_index:self.data_index + self.batch_size]
        for key, value in self.data_states.items():
            batch_states[key] = value[self.data_index:self.data_index + self.batch_size]
        self.data_index += self.batch_size
        if self.data_index >= len(self.data_states["l_img"]) - 1:
            self.data_index = 0
        return batch_actions, batch_states


    def PCA(self, image_array):
        # image_array = image_array.reshape(image_array.shape[0], image_array)
        pca = PCA(self.image_compression_dim)
        image_transformed = pca.fit_transform(image_array)
        print(image_transformed.shape)
        return pca, image_transformed



