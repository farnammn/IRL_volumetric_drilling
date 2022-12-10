from scipy.io import loadmat
from pathlib import Path
import os






data_path = Path(__file__).resolve().parent
data_path = os.path.join(data_path, "data/human_robot_data_p1_train.mat")

annots = loadmat(data_path)
print(annots)

