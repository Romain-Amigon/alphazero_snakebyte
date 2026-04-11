import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inplanes=64, planes=64, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroResNet(nn.Module):
    def __init__(self, board_width=28, board_height=15, channels=4, action_size=4, num_res_blocks=4):
        super(AlphaZeroResNet, self).__init__()

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.resblocks = nn.ModuleList([ResBlock(64, 64) for _ in range(num_res_blocks)])

        # Policy Head
        self.p_conv = nn.Conv2d(64, 2, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        
        p_flat_size = 2 * board_width * board_height
        self.p_fc = nn.Linear(p_flat_size, action_size)

        # Value Head
        self.v_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)

        v_flat_size = 1 * board_width * board_height
        self.v_fc1 = nn.Linear(v_flat_size, 64)
        self.v_fc2 = nn.Linear(64, 1)

        self.action_size = action_size

    def forward(self, x):
        # Allow (C, H, W) or (B, C, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        out = F.relu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            out = block(out)

        # Policy
        p = F.relu(self.p_bn(self.p_conv(out)))
        p = p.flatten(1)
        policy = F.log_softmax(self.p_fc(p), dim=1)

        # Value
        v = F.relu(self.v_bn(self.v_conv(out)))
        v = v.flatten(1)
        
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v))
        return policy, value
