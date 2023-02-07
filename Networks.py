import torch
import torch.nn as nn
import torch.nn.functional as F
#import copy

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup = dict(device=DEVICE, dtype=torch.float)

__all__ =['Net', 'DNN','MNISTClassifier','ResNet','ResNet18']

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

#DNN Fedavg paper used
class DNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 50, 3, padding=1)
        self.fc1 = nn.Linear(50*7*7, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTClassifier(nn.Module):
    """
    Convolutional neural network used in the tutorial for CleverHans.
    This neural network is also used in experiments by Staib et al. (2017) and
    Sinha et al. (2018).
    """

    def __init__(self, nb_filters=64, activation='relu'):
        """
        The parameters in convolutional layers and a fully connected layer are
        initialized using the Glorot/Xavier initialization, which is the
        default initialization method in Keras.
        """

        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=(
            8, 8), stride=(2, 2), padding=(3, 3))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2,
                               kernel_size=(6, 6), stride=(2, 2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(
            nb_filters * 2, nb_filters * 2, kernel_size=(5, 5), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(nb_filters * 2, 10)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.applyActivation(outputs)
        outputs = self.conv2(outputs)
        outputs = self.applyActivation(outputs)
        outputs = self.conv3(outputs)
        outputs = self.applyActivation(outputs)
        outputs = outputs.view((-1, self.num_flat_features(outputs)))
        outputs = self.fc1(outputs)
        # Note that because we use CrosEntropyLoss, which combines
        # nn.LogSoftmax and nn.NLLLoss, we do not need a softmax layer as the
        # last layer.
        return outputs

    def applyActivation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        else:
            raise ValueError("The activation function is not valid.")

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #print(out.size())
        out = self.layer1(out)
        #print(out.size())
        #out = torch.clamp(out, max=3)
        out = self.layer2(out)
        #print(out.size())
        #out = torch.clamp(out, max=3)
        out = self.layer3(out)
        #out = torch.clamp(out, max=3)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)
        #out = torch.clamp(out, max=1.3)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])