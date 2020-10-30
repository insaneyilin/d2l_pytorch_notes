import os
import time
import torch
from torch import nn, optim

import sys
import d2lzh_pytorch as d2l

# check device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

lenet = LeNet()
print(lenet)

# load data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# training function
def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        # iterate during each epoch
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)     # calc loss
            optimizer.zero_grad()  # clear grad. info
            l.backward()           # backward propagation
            optimizer.step()       # optimization
            train_loss_sum += l.cpu().item()    # accumulate training loss
            # accumulate training accurate predicted samples
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]   # accumulate total number of samples during training process
            batch_count += 1  # number of batch counts
        # evaluate after one epoch's training is done
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


# learning rate, number of epochs
lr, num_epochs = 0.001, 5

optimizer = torch.optim.Adam(lenet.parameters(), lr=lr)
train(lenet, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

