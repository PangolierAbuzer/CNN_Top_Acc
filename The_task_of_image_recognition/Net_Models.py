import torch


"""
The CNN model taken from a course on the stepik platform. 
The first network I trained. The accuracy on the test set was 70% after 3 epochs.
I did not test it on the improved training function.

It is based on the standard CNN architecture 
(input layer; blocks including convolution, activation, batch normalization, pooling;
three fully connected layers with optimization; output layer).
"""
class Stepik_CIFAR10_Net(torch.nn.Module):
    def __init__(self):
        super(Stepik_CIFAR10_Net, self).__init__()
        self.batch_norm0 = torch.nn.BatchNorm2d(3)

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.act1 = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.act2 = torch.nn.ReLU()
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.act3 = torch.nn.ReLU()
        self.batch_norm3 = torch.nn.BatchNorm2d(64)

        self.fc1 = torch.nn.Linear(8 * 8 * 64, 256)
        self.act4 = torch.nn.Tanh()
        self.batch_norm4 = torch.nn.BatchNorm1d(256)

        self.fc2 = torch.nn.Linear(256, 64)
        self.act5 = torch.nn.Tanh()
        self.batch_norm5 = torch.nn.BatchNorm1d(64)

        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.batch_norm0(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.batch_norm2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.batch_norm3(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act4(x)
        x = self.batch_norm4(x)

        x = self.fc2(x)
        x = self.act5(x)
        x = self.batch_norm5(x)

        x = self.fc3(x)

        return x


"""
The previous CNN model with my modifications. Replacing two fully connected layers with an avg layer 
and reducing the number of layers. Achieved 76% on the test set in 53 epochs.
No testing was performed on the improved training function.
"""
class My_CIFAR10_Net_First(torch.nn.Module):
    def __init__(self):
        super(My_CIFAR10_Net_First, self).__init__()
        self.batch_norm0 = torch.nn.BatchNorm2d(3)

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(64)
        self.act3 = torch.nn.ReLU()

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.batch_norm0(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.act3(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


"""
My second CNN model. The first prototype of this model recognized 76% of the images in the test set. 
By removing the first pooling layer, I achieved 78%. After adjusting the batch size and manual step, 
I achieved an accuracy of ~84%. The final version of the training function allowed the model to recognize
85.5% of the images in the test set in 26 epochs. I assume that the augmented version of the function
will increase the accuracy to 89%.

Expanded the number of layers. Added a dropout method for the fully connected layer and convolution layers.
"""
class My_CIFAR10_Net_Second(torch.nn.Module):
    def __init__(self):
        super(My_CIFAR10_Net_Second, self).__init__()
        self.batch_norm0 = torch.nn.BatchNorm2d(3)

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        self.act1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(64)
        self.act3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(2, 2)

        self.conv4 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.batch_norm4 = torch.nn.BatchNorm2d(128)
        self.act4 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(2, 2)
        self.dropout2d_4 = torch.nn.Dropout2d(p=0.2)

        self.conv5 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.batch_norm5 = torch.nn.BatchNorm2d(256)
        self.act5 = torch.nn.ReLU()
        self.pool5 = torch.nn.MaxPool2d(2, 2)
        self.dropout2d_5 = torch.nn.Dropout2d(p=0.2)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = torch.nn.Dropout(p=0.4)

        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.batch_norm0(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.act4(x)
        x = self.pool4(x)
        x = self.dropout2d_4(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.act5(x)
        x = self.pool5(x)
        x = self.dropout2d_5(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc(x)

        return x


"""
The third model. Expanded data coverage. 640 channels at the output. High computational load. 
With maximum optimization of the training function, the accuracy reached 87.16% in 45 epochs. 
Adding augmentation, increased the recognition accuracy on the test set to 91.06% in ~95 epochs. 
The most powerful model presented.
"""
class My_CIFAR10_Net_Third(torch.nn.Module):
    def __init__(self):
        super(My_CIFAR10_Net_Third, self).__init__()
        self.batch_norm0 = torch.nn.BatchNorm2d(3)

        self.conv1 = torch.nn.Conv2d(3, 20, 3, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(20)
        self.act1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(20, 40, 3, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(40)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(40, 80, 3, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(80)
        self.act3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(2, 2)

        self.conv4 = torch.nn.Conv2d(80, 160, 3, padding=1)
        self.batch_norm4 = torch.nn.BatchNorm2d(160)
        self.act4 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(2, 2)
        self.dropout2d_4 = torch.nn.Dropout2d(p=0.2)

        self.conv5 = torch.nn.Conv2d(160, 320, 3, padding=1)
        self.batch_norm5 = torch.nn.BatchNorm2d(320)
        self.act5 = torch.nn.ReLU()
        self.pool5 = torch.nn.MaxPool2d(2, 2)
        self.dropout2d_5 = torch.nn.Dropout2d(p=0.2)

        self.conv6 = torch.nn.Conv2d(320, 640, 3, padding=1)
        self.batch_norm6 = torch.nn.BatchNorm2d(640)
        self.act6 = torch.nn.ReLU()

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = torch.nn.Dropout(p=0.4)

        self.fc = torch.nn.Linear(640, 10)

    def forward(self, x):
        x = self.batch_norm0(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.act4(x)
        x = self.pool4(x)
        x = self.dropout2d_4(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.act5(x)
        x = self.pool5(x)
        x = self.dropout2d_5(x)

        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = self.act6(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc(x)

        return x

