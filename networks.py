import torch
import config

class GENERATOR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_sequence = torch.nn.Sequential(
                torch.nn.Linear(10, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, config.WIDTH*config.HEIGHT),
                #torch.nn.Tanh()
                torch.nn.Sigmoid()
                )

    def forward(self, x):
        return self.linear_relu_sequence(x).view(64, 64)

class DISCRIMINATOR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptron = torch.nn.Sequential(
                torch.nn.Linear(config.WIDTH*config.HEIGHT, config.WIDTH*config.HEIGHT),
                torch.nn.Linear(config.WIDTH*config.HEIGHT, 1),
                torch.nn.Sigmoid())

    def forward(self, x):
        return self.perceptron(x.view(-1))

###########################################################################################################################

class GENERATOR_CONV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.DECONV = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(100, 8, kernel_size=4, stride=1, padding=0, bias=False),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
                torch.nn.Tanh()
                )

    def forward(self, x):
        return self.DECONV(x)

class DISCRIMINATOR_CONV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CONV = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(8, 16, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(16, 32, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(64, 1, 4, 2, 0, bias=False),
                torch.nn.Sigmoid()
                )

    def forward(self, x):
        return self.CONV(x).view(-1)
