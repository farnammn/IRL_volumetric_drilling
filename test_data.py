from scipy.io import loadmat
from pathlib import Path
import os
import numpy as np
from sklearn.cluster import KMeans
import copy
import pickle



"""
'x_f', 'vx_f', 'ax_f', 'x_l', 'ay_f', 'vy_f', 'y_l', 'y_f', 'dt'
"""


def load_data(main_path, file_path):
    return loadmat(os.path.join(data_path,"data/" + file_path))

data_path = Path(__file__).resolve().parent

data_processed = load_data(data_path, "human_robot_data_p1_train.mat")


print(data_processed.keys())

#constants
l = 3.476
x_follower = data_processed["x_f"]
vx_follower = data_processed["vx_f"]
ax_follower = data_processed["ax_f"]
x_leader = data_processed["x_l"]
ay_follower = data_processed["ay_f"]
vy_follower = data_processed["vy_f"]
y_leader = data_processed["x_f"]
y_follower = data_processed["y_f"]
dt = data_processed["dt"]


#################
def create_state(x_f, vx_f, ax_f, x_l, ay_f, vy_f, y_l, y_f, dt, l):
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


def descretize(action_space, n_c=2):
    """desceritize the action space by kmeans clustering
    in other words, it maps each action to its center.
    input: n_c --> number of clusters
    action space: should be in shape n_samples * 2
    output: descretized action_space

    """
    aux = copy.deepcopy(action_space)
    kmeans = KMeans(n_clusters=n_c, random_state=0).fit(aux.reshape(-1, 2))
    centers = kmeans.cluster_centers_
    print(centers.shape)
    lab = kmeans.labels_
    for i in range(lab.shape[0]):
        k = lab[i]
        action_space[0, i] = centers[k, 0]
        action_space[1, i] = centers[k, 1]
    return action_space


def cost_f(x_f, y_f, theta_f, x_l, y_l, u_a, r_1 = 0.05, r_2 = 1., r_3 = 0.1,r_4 = 1.0, r_5 = 0.1, r_6 = 0.5  ):
    """
    calculate cost fucntion
    """
    x_rel = x_l - x_f
    y_rel = y_l - y_f
#     v_rel =
    f1 = (x_rel > 2.5)*(np.log(1+np.exp(r_1*(x_rel-2.5))) - np.log(2) )#not too far from leader
    f2 = (x_rel < 2.5)*( np.log(1+np.exp(-r_2*(x_rel-2.5))) -np.log(2) ) # leader should remain first
#     f3 =
#     f4 =
    f5 = np.log(1+np.exp(r_5*np.abs(y_rel)))-np.log(2)
    f6 = (y_f > 2)*( np.log(1+np.exp(r_6*(y_f-2.))) - np.log(2) ) + (y_f < -2)*( np.log(1+np.exp(-r_6*(y_f+2.))) - np.log(2) )
    f = f1+f2+f5+f6
    return f.reshape(1,-1)


def creat_dataset(x_f, vx_f, ax_f, x_l, ay_f, vy_f, y_l, y_f, dt, l):
    state, action = create_state(x_f, vx_f, ax_f, x_l, ay_f, vy_f, y_l, y_f, dt, l)
    action = descretize(action, n_c=15)
    cost_p = cost_f(state[0, :], state[1, :], state[2, :], state[1, :], state[1, :], action[0, :], r_1=0.05, r_2=1.,
                    r_3=0.1, r_4=1.0, r_5=0.1, r_6=0.5)
    return state, action, cost_p
a,b,c = creat_dataset(x_follower, vx_follower, ax_follower, x_leader, ay_follower, vy_follower, y_leader, y_follower, 0.0167, l)


# Building trajectory
# chunk the wole data to different trajectories
def make_trajectory(state_sp, action_sp, cost_p, w_s=30):
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
    dict_tra = {}
    dict_tra["state"] = traj_s
    dict_tra["action"] = traj_s
    dict_tra["cost"] = traj_s

    return dict_tra
trajectory_dictionary = make_trajectory(a, b, c, w_s=30)
with open(os.path.join(data_path,"data/" + 'data_traj.pickle'), 'wb') as handle:
    pickle.dump(trajectory_dictionary, handle)


