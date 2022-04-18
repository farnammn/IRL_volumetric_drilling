import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def network_params_names(network):
    names = []
    named_params = []
    for module in network.modules():
        for name, param in module.named_parameters():
            if name in names:
                continue
            names.append(name)
            named_params.append((name, param))
    return names, named_params


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.leaky_relu):
        super(FCBody, self).__init__()
        if len(hidden_units) > 0:
            dims = (state_dim,) + hidden_units

            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

            self.gate = gate
            self.feature_dim = dims[-1]
        else:
            self.feature_dim = state_dim
            self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class ConvolutionalBody(nn.Module):
    def __init__(self, state_dim, hidden_units=((64, 64), (64, 64)), kernels=(5, 5), strides=(1, 1),
                 paddings=(0, 0), pool=None, pool_kernels=None, pool_strides=None, gate=F.leaky_relu):
        super(ConvolutionalBody, self).__init__()
        if len(hidden_units) > 0:
            dims = (state_dim,) + hidden_units
            self.layers = []
            if pool == None:
                for dim_in, dim_out, kernel, stride, padding in zip(dims[:-1], dims[1:], kernels, strides, paddings):
                    self.layers.append(nn.Conv2d(dim_in, dim_out, kernel, stride=stride, padding=padding))
            else:
                for dim_in, dim_out, kernel, stride, padding, pool_kernel, pool_stride in zip(dims[:-1], dims[1:], kernels, strides, paddings, pool_kernels, pool_strides):
                    self.layers.append(nn.Conv2d(dim_in, dim_out, kernel, stride=stride, padding=padding))
                    self.layers.append(pool(pool_kernel, stride=pool_stride))
            self.layers = nn.ModuleList(self.layers)
            self.gate = gate
            self.feature_dim = dims[-1]
        else:
            self.feature_dim = state_dim
            self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class VolumetricValueNet(nn.Module):
    def __init__(self, config):
        super(VolumetricValueNet, self).__init__()
        self.l_img_body = ConvolutionalBody(config.img_dim, config.convolutional_hidden_units, config.convlutional_kernels,
                                       config.convlutional_strides, config.convlutional_padings, config.pooling_layer,
                                       config.pooling_kernels, config.pool_strides, gate=config.convlutional_gate)
        self.r_img_body = ConvolutionalBody(config.img_dim, config.convolutional_hidden_units, config.convlutional_kernels,
                                       config.convlutional_strides, config.convlutional_padings, config.pooling_layer,
                                       config.pooling_kernels, config.pool_strides, gate=config.convlutional_gate)
        self.depth_body = ConvolutionalBody(config.img_dim, config.convolutional_hidden_units, config.convlutional_kernels,
                                       config.convlutional_strides, config.convlutional_padings, config.pooling_layer,
                                       config.pooling_kernels, config.pool_strides, gate=config.convlutional_gate)
        self.segm_body = ConvolutionalBody(config.img_dim, config.convolutional_hidden_units, config.convlutional_kernels,
                                      config.convlutional_strides, config.convlutional_padings, config.pooling_layer,
                                      config.pooling_kernels, config.pool_strides, gate=config.convlutional_gate)
        self.voxel_removed_body = ConvolutionalBody(config.img_dim, config.convolutional_hidden_units, config.convlutional_kernels,
                                      config.convlutional_strides, config.convlutional_padings, config.pooling_layer,
                                      config.pooling_kernels, config.pool_strides, gate=config.convlutional_gate)
        self.voxel_color_body = ConvolutionalBody(config.img_dim, config.convolutional_hidden_units, config.convlutional_kernels,
                                      config.convlutional_strides, config.convlutional_padings, config.pooling_layer,
                                      config.pooling_kernels, config.pool_strides, gate=config.convlutional_gate)

        self.pose_cam_body = FCBody(config.pose_cam_dim, config.FC_hidden_units, gate=config.FC_gate)
        self.pose_drill_body = FCBody(config.pose_drill_dim, config.FC_hidden_units, gate=config.FC_gate)

        self.fc_l_img = FCBody(torch.prod(self.l_img_body.feature_dim), config.img_FC_hidden_units)
        self.fc_r_img = FCBody(torch.prod(self.r_img_body.feature_dim), config.img_FC_hidden_units)
        self.fc_depth = FCBody(torch.prod(self.depth_body.feature_dim), config.img_FC_hidden_units)
        self.fc_segm = FCBody(torch.prod(self.segm_body.feature_dim), config.img_FC_hidden_units)
        self.fc_voxel_removed = FCBody(torch.prod(self.voxel_removed_body.feature_dim), config.img_FC_hidden_units)
        self.fc_voxel_color = FCBody(torch.prod(self.voxel_color_body.feature_dim), config.img_FC_hidden_units)

        self.fc = nn.Linear(config.state_presentation_dim + config.action_dim, config.N)

    def forward(self, l_img, r_img, segm, depth, voxel_removed, voxel_color, pose_cam, pose_drill, action):
        l_img_pres = self.fc_l_img(self.l_img_body(l_img))
        r_img_pres = self.fc_r_img(self.r_img_body(r_img))
        depth_pres = self.fc_depth(self.depth_body(depth))
        segm_pres = self.fc_segm(self.segm_body(segm))
        voxel_removed_pres = self.fc_voxel_removed(self.voxel_removed_body(voxel_removed))
        voxel_color_pres = self.fc_voxel_color(self.voxel_color_body(voxel_color))

        pose_cam_pres = self.pose_cam_body(pose_cam)
        pose_drill_pres = self.pose_drill_body(pose_drill)

        state_pres = l_img_pres + r_img_pres + depth_pres + segm_pres + voxel_removed_pres + voxel_color_pres \
                     + pose_cam_pres + pose_drill_pres

        return self.fc(torch.cat(state_pres, action))