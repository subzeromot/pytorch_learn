import os
import math
import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from lenet import LeNet
from densenet import densenet_cifar

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Process:
    def __init__(self, _net):
        # Transform function
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load dataset
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        train_set = torchvision.datasets.CIFAR10('./datasets', train=True, download=True, transform=self.transform)
        test_set = torchvision.datasets.CIFAR10('./datasets', train=True, download=False, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(device)

        # Load NET
        self.net = _net
        # print(net)

        if device == 'cuda':
            self.net = nn.DataParallel(self.net)
            torch.backends.cudnn.benchmark = True

        lr = 0.01 #1e-1
        momentum = 0.9
        weight_decay = 1e-4

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 225])
    
    def scheduler_step(self):
        self.scheduler.step()

    # Training
    def train(self, epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print("Epoch:" , epoch, " ========= loss: ", loss)
        acc = 100.*correct/total
        loss = train_loss / batch_idx
        print('Epoch: %d, train loss: %.6f, acc: %.3f%% (%d/%d)' % (epoch, loss, acc, correct, total))
        return loss

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        loss = test_loss / batch_idx
        print('Epoch: %d, test loss: %.6f, acc: %.3f%% (%d/%d)' % (epoch, loss, acc, correct, total))
        return loss

    def plot_loss(self, train_losses, test_losses):
        print(train_losses)
        print(test_losses)
        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(range(len(test_losses)), test_losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.show()
        plt.draw()
        plt.savefig('result.png')

def main():
    ### DenseNet
    # net = densenet_cifar().to(device)

    ### LeNET
    net = LeNet().to(device)

    process = Process(net)
    num_epoch = 2

    train_losses = []
    test_losses = []

    for epoch in range(num_epoch):
        process.scheduler_step()
        train_loss = process.train(epoch)
        test_loss = process.test(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        start_epoch = epoch

if __name__ == "__main__":
    main()