from network import VolumetricValueNet
import torch
from torch.autograd import Variable
import numpy as np
import random
from utils import plot_mu
from logger import get_logger
import os
from os import mkdir

def irl(data_list, Rs, config, is_rew_fix=True):
    logger = get_logger(log_dir=config.log_dir, log_level=config.log_level)

    def record_online_data(data, name, total_steps, offset=0):
        path_name = config.log_dir + name + "/"
        if not os.path.exists(path_name):
            mkdir(path_name)
        with open(path_name + '.txt', 'a') as file:
            file.write(str(data) + '\n')
        logger.add_scalar(name, data, total_steps + offset)
        logger.info('steps %d, %s %s' % (total_steps + offset, name, data))

    mu = Variable(torch.randn(len(data_list)).type(torch.FloatTensor), requires_grad=True)
    mu = torch.absolute(mu) / torch.sum(mu)
    if not is_rew_fix:
        w = Variable(torch.randn(config.state_dim + config.action_dim).type(torch.FloatTensor), requires_grad=True)
    VaRs = Rs

    for total_steps in config.num_steps:
        loss = 0
        if not is_rew_fix:
            optimizer = torch.optim.Adam([w, mu], lr=config.lr)
            rewards = torch.zeros(len(data_list))
            for j, data in enumerate(data_list):
                phi = torch.concat((data["state_rep"], data["actions_rep"]), dim=0)
                rewards_temp = torch.matmul(phi, w)
                VaRs[j] = torch.sum(rewards_temp)

                rewards_temp = rewards_temp.multinomial(num_samples=config.batch_size, replacement=False)
                rewards[j] = torch.sum(rewards_temp) / len(rewards_temp)
        else:
            optimizer = torch.optim.Adam(mu, lr=config.lr)

        for j, data in enumerate(data_list):
            for i, var in enumerate(VaRs[:(j+1)]):
                if is_rew_fix:
                    rewards = random.sample(data["reward"], config.batch_size)

                loss += mu[i] * (rewards[j] - var / len(data["state_rep"])) / (len(data_list)-i)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mu = torch.absolute(mu) / torch.sum(mu)
        record_online_data(data=loss, total_steps=total_steps, name="loss")
        record_online_data(data=mu, total_steps=total_steps, name="mu")
        record_online_data(data=w[-config.action_dim:-1], total_steps=total_steps, name="w")

    plot_mu(config.log_dir, w[-config.action_dim:-1])


