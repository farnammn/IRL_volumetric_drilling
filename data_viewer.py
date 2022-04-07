from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt
import numpy as np


def pose_to_matrix(pose):
    quat_norm = np.linalg.norm(pose[:, 3:], axis=-1)
    assert np.all(np.isclose(quat_norm, 1.0))
    r = R.from_quat(pose[:, 3:]).as_matrix()
    t = pose[:, :3]
    tau = np.identity(4)[None].repeat(pose.shape[0], axis=0)
    tau[:, :3, :3] = r
    tau[:, :3, -1] = t

    return tau

def view_data():
    for i in range(l_img.shape[0]):
        plt.subplot(221)
        plt.imshow(l_img[i])
        plt.subplot(222)
        plt.imshow(r_img[i])
        plt.subplot(223)
        plt.imshow(depth[i], vmax=1)
        plt.subplot(224)
        plt.imshow(segm[i])

        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default=None)
    args = parser.parse_args()

    if args.file is not None:
        file = h5py.File(args.file, 'r')
        l_img = file["data"]["l_img"][()]
        r_img = file["data"]["r_img"][()]
        depth = file["data"]["depth"][()]
        segm = file["data"]["segm"][()]
        K = file['metadata']["camera_intrinsic"][()]
        extrinsic = file['metadata']['camera_extrinsic'][()]

        pose_cam = pose_to_matrix(file['data']['pose_main_camera'][()])
        pose_cam = np.matmul(pose_cam, np.linalg.inv(extrinsic)[None])  # update pose so world directly maps to CV
        pose_drill = pose_to_matrix(file['data']['pose_mastoidectomy_drill'][()])

        view_data()
