import torch
import torch.nn as nn



class SimplePatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=1, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(input_channels, ndf, 4, 2, 1),
                  nn.LeakyReLU(0.2)]


        for i in range(1, n_layers):
            in_c = ndf * min(2 ** (i - 1), 8)
            out_c = ndf * min(2 ** i, 8)
            layers += [
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2)
            ]

        layers += [nn.Conv2d(out_c, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # 输入标准化
        x = (x - x.mean()) / (x.std() + 1e-8)
        return self.model(x)