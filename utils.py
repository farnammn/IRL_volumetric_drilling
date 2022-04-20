import matplotlib.pyplot as plt
import numpy as np
import os
from os import mkdir
import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class myArrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def get_time_str():
    return datetime.datetime.now().strftime("%d-%b_%H-%M-%S")


def plot_mu(path, mu):
    x = [i for i in range(len(mu))]
    plt.bar(x, mu)
    plt.xlabel("alpha = i/N")
    plt.ylabel("mu")
    plt.title("Estimated mu")
    plt.legend()
    plt.grid()
    plt.savefig(path + "final-mu.png")
    plt.close()


def plot_images(path, image, reward, i):
    plt.savefig(path + "image_" + str(i), image)
    with open(path + "reward_"+ str(i), "a") as file:
        file.writelines(reward)


def mkdirs(path):
    os.makedirs(path)
    os.chmod(path, 0o755)



def read_log(path):
    logs = {}
    current, dirs, files = os.walk(path).__next__()
    for dir in dirs:
        logs[dir] = []
        run_log = []
        min_length = np.inf
        new_path = path + dir + "/"
        for filename in os.listdir(new_path):
            if ".txt" in filename:
                file_log = []
                with open(new_path + filename, "r") as f:
                    for line in f:
                        ret = float(line.strip())
                        file_log.append(ret)
                    if len(file_log) < min_length:
                        min_length = len(file_log)
                run_log.append(file_log)
            for log in run_log:
                logs[dir].append(log[:min_length])
    return logs


def plot_log(name, log, color, shadow):
    '''
    :param name: name to be used in the legend of figure
    :param log: data to be plotted
    :param color:
    :param shadow: "SE" standard error, "STD" standard deviation
    :return:
    '''
    mean = np.mean(log, axis=0)
    std = np.std(log, axis=0)
    plt.plot(mean, color, label=str(name))
    if shadow == "SE":
        std_error = std / (len(log) ** 0.5)
    elif shadow == "STD":
        std_error = std
    plt.fill_between(range(len(log[0])), mean - std_error, mean + std_error, color=color, alpha=0.3)


def plot(data, title):
    '''
    :param data: list of tuples (path, legend) of directories containing the data you want to plot in one figure
                if there are more than one file in each path containing the name "metric" it plots average of them
                with shadow showing standard error (you can change shadow to "STD" to plot standard deviation)
    :param metric: a keyword in the name of the files in each directory containing desired data to be plotted
    :param title: title of the final figure
    :return:
    '''

    for i in range(len(data)):
        path, name = data[i]
        logs = read_log(path)

        for metric, log in logs.items():
            if len(log) == 0:
                continue
            plot_log(name, log, color="C" + str(i), shadow="SE")
            plt.xlabel('Episode')
            plt.ylabel(metric)
            plt.title(title)
            plt.legend()
            plt.grid()
            plt.savefig(path + metric+'.png')
            plt.close()