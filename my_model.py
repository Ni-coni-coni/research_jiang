import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class CNN(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc_linear = nn.Linear(32 * 9 * 9, 256)

        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc_linear(x.view(-1, 32 * 9 * 9)))
        return self.actor_linear(x)

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]

    def pull(self, model):
        self.load_state_dict(model.state_dict())