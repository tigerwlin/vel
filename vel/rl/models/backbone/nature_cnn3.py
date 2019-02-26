"""
Code based loosely on implementation:
https://github.com/openai/baselines/blob/master/baselines/ppo2/policies.py

Under MIT license.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import vel.util.network as net_util

from vel.api.base import LinearBackboneModel, ModelFactory


class NatureCnn3(LinearBackboneModel):
    """ Neural network as defined in the paper 'Human-level control through deep reinforcement learning' """
    def __init__(self, input_width, input_height, input_channels, output_dim=512):
        super().__init__()

        self._output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(8, 8),
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4),
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1
        )

        self.linear1 = nn.Linear(2, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(1, 1024)
        self.linear4 = nn.Linear(1024, 512)

        self.final_width = net_util.convolutional_layer_series(input_width, [
            (8, 0, 2),
            (4, 0, 2),
            (3, 0, 1)
        ])

        self.final_height = net_util.convolutional_layer_series(input_height, [
            (8, 0, 2),
            (4, 0, 2),
            (3, 0, 1)
        ])

        self.linear_layer1 = nn.Linear(
            self.final_width * self.final_height * 64 + 512 + 512,  # 64 is the number of channels of the last conv layer
            1024
        )
        self.linear_layer2 = nn.Linear(1024, self.output_dim)

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, image):
        input1 = image['obs']['environment']
        input2 = image['obs']['goal'].float()
        input3 = image['action'].float()

        result1 = input1.permute(0, 3, 1, 2).contiguous().type(torch.float) / 255.0
        result1 = F.relu(self.conv1(result1))
        result1 = F.relu(self.conv2(result1))
        result1 = F.relu(self.conv3(result1))

        result2 = input2.view(input2.size(0), -1)
        # result2 = torch.cat((result2, input3), 1)
        result2 = F.leaky_relu(self.linear1(result2))
        result2 = F.leaky_relu(self.linear2(result2))

        result3 = input3.view(input3.size(0), -1)
        result3 = F.leaky_relu(self.linear3(result3))
        result3 = F.leaky_relu(self.linear4(result3))

        flattened1 = result1.view(result1.size(0), -1)
        flattened2 = result2.view(result2.size(0), -1)
        flattened3 = result3.view(result3.size(0), -1)
        flattened = torch.cat((flattened1, flattened2, flattened3), 1)

        result = F.leaky_relu(self.linear_layer1(flattened))
        result = F.leaky_relu(self.linear_layer2(result))

        return result


def create(input_width, input_height, input_channels=1, output_dim=512):
    def instantiate(**_):
        return NatureCnn3(
            input_width=input_width, input_height=input_height, input_channels=input_channels,
            output_dim=output_dim
        )

    return ModelFactory.generic(instantiate)


# Add this to make nicer scripting interface
NatureCnnFactory3 = create

