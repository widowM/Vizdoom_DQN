import torch.nn as nn
import torch.nn.functional as F

# The network of the Health Gathering Supreme scenario
class ConvNet2(nn.Module):
    def __init__(self, w=45, h=30, init_zeros=False, stack_size=1, out=3):
        """
        Description
        ---------------
        Constructor of Deep Q-network class.

        Parameters
        ---------------
        w          : Int, input width (default=120)
        h          : Int, input height (default=160)
        init_zeros : Boolean, whether to initialize the weights to zero or not.
        stack_size : Int, input dimension which is the number of frames to stack to create motion (default=4)
        out        : Int, the number of output units, it corresponds to the number of possible actions (default=3).
                     Be careful, it must be changed when considering a different number of possible actions.
        """

        super(ConvNet2, self).__init__()

        # Conv Module
        self.conv_1 = nn.Conv2d(in_channels=stack_size, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        #         self.conv_3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2)
        if init_zeros:
            nn.init.constant_(self.conv_1.weight, 0.0)
            nn.init.constant_(self.conv_2.weight, 0.0)
            nn.init.constant_(self.conv_3.weight, 0.0)

        convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)  # width of last conv output
        convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2)  # height of last conv output
        linear_input_size = convw * convh * 64

        self.fc = nn.Linear(linear_input_size, 512)
        self.output = nn.Linear(512, out)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        #         x = F.relu(self.conv_3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.output(x)

# The network of the Basic scenario
class ConvNet(nn.Module):
    def __init__(self, action_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*8, 120) #16 4 8 | 16 12 19
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_size)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*4*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def conv2d_size_out(size, kernel_size=5, stride=2):
    """
    Description
    --------------
    Compute the output dimension when applying a convolutional layer.

    Parameters
    --------------
    size        : Int, width or height of the input.
    kernel_size : Int, the kernel size of the conv layer (default=5)
    stride      : Int, the stride used in the conv layer (default=2)
    """

    return (size - (kernel_size - 1) - 1) // stride + 1