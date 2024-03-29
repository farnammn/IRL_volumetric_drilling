import torch
import random
# from utils import plot_mu, plot_images
from logger import get_logger
import os
from os import mkdir
import numpy as np
from torch.autograd import Variable

def irl(data_list, config, is_rew_fix=False, w=None):
    logger = get_logger(log_dir=config.log_dir, log_level=config.log_level)

    def record_online_data(data, name, total_steps, offset=0):
        path_name = config.log_dir + name + "/"
        if not os.path.exists(path_name):
            mkdir(path_name)
        with open(path_name + '.txt', 'a') as file:
            file.write(str(data) + '\n')
        logger.add_scalar(name, data, total_steps + offset)
        logger.info('steps %d, %s %s' % (total_steps + offset, name, data))

    if is_rew_fix:
        VaRs = []
        for data in data_list:
            VaRs.append(np.sum(data["reward"]))

        data_list = [data for _, data in sorted(zip(VaRs, data_list), key=lambda pair: pair[0])]
        VaRs = sorted(VaRs)
    mu = torch.rand(len(data_list))
    mu = torch.absolute(mu) / torch.sum(mu)
    mu.requires_grad = True

    for total_steps in range(config.num_steps):
        loss = 0
        if is_rew_fix:
            optimizer = torch.optim.Adam([mu], lr=config.lr)
        if not is_rew_fix:
            w = torch.rand(config.state_dim + config.action_dim, dtype=torch.float64)
            optimizer = torch.optim.Adam([w, mu], lr=config.lr)
            # maybe should be out of loop
        if not is_rew_fix:
            rewards = torch.zeros(len(data_list))
            VaRs = []
            for j, data in enumerate(data_list):
                # print(np.array(data["state_rep"]).shape)
                # print(np.array(data["action_rep"]).shape)
                # print(len(data["state_rep"]))
                print(type(data))
                phi = torch.concat((torch.tensor(data["state_rep"]), torch.tensor(data["action_rep"])), dim=1)
                rewards_temp = torch.matmul(phi, w)
                # print(rewards_temp)
                VaRs.append(torch.sum(rewards_temp))

                # rewards_temp = rewards_temp.multinomial(num_samples=config.batch_size, replacement=False)
                rewards[j] = torch.sum(rewards_temp) / len(rewards_temp)
            data_list = [data for _, data in sorted(zip(VaRs, data_list), key=lambda pair: pair[0])]
            rewards = [reward for _, reward in sorted(zip(VaRs, rewards), key=lambda pair: pair[0])]

        for j, data in enumerate(data_list):
            for i, var in enumerate(VaRs[:(j+1)]):
                if is_rew_fix:
                    # rewards = np.random.choice(data["reward"], config.batch_size)
                    print("kire khar")
                    rewards = data["reward"]
                if len(data["state_rep"]) > 0:
                    v = var / len(data["state_rep"])
                else:
                    v = 0
                # print(mu[i])
                # print(j)
                # print(len(data_list))
                # print(rewards[j])

                loss += mu[i] * (rewards[j] - v) / (len(data_list) - i + 1)


        optimizer.zero_grad()
        print("hi")
        loss.backward()
        optimizer.step()
        mu = torch.absolute(mu.detach()) / torch.sum(mu.detach())
        mu.requires_grad = True
        record_online_data(data=loss, total_steps=total_steps, name="loss")
        if not is_rew_fix:
            idxs = [-8, -7, -6]
            for i, idx in enumerate(idxs):
                record_online_data(data=w[idx], total_steps=total_steps, name="w_"+str(i))
        for i in range(len(data_list)):
            record_online_data(data=mu[i], total_steps=total_steps, name="mu_"+str(i))

    # plot_mu(config.log_dir, mu)

def irl_dr(data_list, config, is_rew_fix=False, w=None):
    logger = get_logger(log_dir=config.log_dir, log_level=config.log_level)


    def record_online_data(data, name, total_steps, offset=0):
        path_name = config.log_dir + name + "/"
        if not os.path.exists(path_name):
            mkdir(path_name)
        with open(path_name + '.txt', 'a') as file:
            file.write(str(data) + '\n')
        logger.add_scalar(name, data, total_steps + offset)
        logger.info('steps %d, %s %s' % (total_steps + offset, name, data))



    if is_rew_fix:
        VaRs = []
        for data in data_list:
            VaRs.append(np.sum(data["reward"]))

        data_list = [data for _, data in sorted(zip(VaRs, data_list), key=lambda pair: pair[0])]
        VaRs = sorted(VaRs)
    mu = torch.rand(len(data_list))
    mu = torch.absolute(mu) / torch.sum(mu)
    mu.requires_grad = True


    for total_steps in range(config.num_steps):
        loss = 0
        if is_rew_fix:
            optimizer = torch.optim.Adam([mu], lr=config.lr)
        if not is_rew_fix:
            w = torch.rand(7, dtype=torch.float64)
            optimizer = torch.optim.Adam([w, mu], lr=config.lr)
            # maybe should be out of loop
        if not is_rew_fix:
            rewards = torch.zeros(len(data_list))
            VaRs = []

            for j, data in enumerate(data_list):
                # print(np.array(data["state_rep"]).shape)
                # print(np.array(data["action_rep"]).shape)
                # print(len(data["state_rep"]))
                phi = torch.concat((torch.tensor(data["state_rep"]), torch.tensor(data["action_rep"])), dim=1)
                print("Show me the shape for data",phi.shape)
                print("Show me the shape for data", w.shape)
                rewards_temp = torch.matmul(phi, w)
                # print(rewards_temp)

                VaRs.append(torch.sum(rewards_temp))

                # rewards_temp = rewards_temp.multinomial(num_samples=config.batch_size, replacement=False)
                rewards[j] = torch.sum(rewards_temp) / len(rewards_temp)
            data_list = [data for _, data in sorted(zip(VaRs, data_list), key=lambda pair: pair[0])]
            rewards = [reward for _, reward in sorted(zip(VaRs, rewards), key=lambda pair: pair[0])]

        for j, data in enumerate(data_list):
            for i, var in enumerate(VaRs[:(j+1)]):
                if is_rew_fix:
                    # rewards = np.random.choice(data["reward"], config.batch_size)
                    print("kire khar")
                    rewards = data["reward"]
                if len(data["state_rep"]) > 0:
                    v = var / len(data["state_rep"])
                else:
                    v = 0
                # print(mu[i])
                # print(j)
                # print(len(data_list))
                # print(rewards[j])

                loss += mu[i] * (rewards[j] - v) / (len(data_list) - i + 1)


        optimizer.zero_grad()
        print("hi")
        loss.backward()
        optimizer.step()
        mu = torch.absolute(mu.detach()) / torch.sum(mu.detach())
        mu.requires_grad = True
        record_online_data(data=loss, total_steps=total_steps, name="loss")
        if not is_rew_fix:
            idxs = [-5, -4, -3]
            for i, idx in enumerate(idxs):
                record_online_data(data=w[idx], total_steps=total_steps, name="w_"+str(i))
        for i in range(len(data_list)):
            record_online_data(data=mu[i], total_steps=total_steps, name="mu_"+str(i))

