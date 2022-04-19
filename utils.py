import matplotlib.pyplot as plt
import numpy as np
import os
from os import mkdir
import datetime


def get_time_str():
    return datetime.datetime.now().strftime("%d-%b_%H-%M-%S")


def plot_mu(path, mu):
    mkdir(path)
    x = [i for i in range(len(mu))]
    plt.bar(x, mu)
    plt.xlabel("alpha = i / N")
    plt.ylabel("mu")
    plt.title("Estimated mu")
    plt.legend()
    plt.grid()
    plt.savefig(path + "final-mu.png")
    plt.close()

def plot_w(path, mu):
    pass

def mkdirs(path):
    os.makedirs(path)
    os.chmod(path, 0o755)

