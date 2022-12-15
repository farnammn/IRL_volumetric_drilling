from scipy.io import loadmat
from pathlib import Path
import os



"""
'x_f', 'vx_f', 'ax_f', 'x_l', 'ay_f', 'vy_f', 'y_l', 'y_f', 'dt'
"""


def load_data(main_path, file_path):
    return loadmat(os.path.join(data_path,"data/" + file_path))

data_path = Path(__file__).resolve().parent
# data_path = os.path.join(data_path, "data/human_robot_data_p1_train.mat")
data_processed = load_data(data_path, "robot_data_10Hz_p1_train.mat")


# annots = loadmat(data_path)

print(data_processed.keys())
print(data_processed["state_robot"].shape)
print(data_processed["state_robot"][2,:])

# '__globals__', 'k_delay'