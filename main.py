
import json
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils

import os
import argparse
from utils import check_dataset


parser = argparse.ArgumentParser(description='PyTorch Challenge Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--checkpoint', default=None,
                    type=str, help='dir to checkpoint')
parser.add_argument('--epochs', '-e', default=200,
                    type=int, help='num of epochs')
parser.add_argument('--batch_size', '-b', default=32,
                    type=int, help='batch size')
parser.add_argument('--pretrained', '-p', action='store_true',
                    help='using pretrained model')
parser.add_argument('--lr_decay_epoch', default=60, type=int,
                    help='Specify the epoch when lr will be decayed')
parser.add_argument('--cuda', action='store_true',
                    help='using cuda')
parser.add_argument('--seed', default=2019, type=int, help='seed number')
parser.add_argument('--model_arch', default='vgg19',
                    type=str, help='Predefined model: vgg16 or vgg19')
parser.add_argument('--checkpoint_postfix', default='2019',
                    type=str, help='Checkpoint file name postfix')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

ROOT_DIR = os.getcwd()
args.dataset_dir = '/datasets/flower_data'
args.train_dir = ROOT_DIR + args.dataset_dir + '/train'
args.valid_dir = ROOT_DIR + args.dataset_dir + '/valid'
args.checkpoint_dir = '{}{}{}_{}.pt'.format(
    ROOT_DIR, args.output_dir, args.model_arch, args.checkpoint_postfix)

args.device = torch.device(
    "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


train_dataset = check_dataset(args.train_dir)
valid_dataset = check_dataset(args.valid_dir, training=False)

train_loader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
test_loader = data.DataLoader(
    valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

# Load label

with open(ROOT_DIR + '/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

n_classes = len(cat_to_name.keys())
print('Number of classes: ' + str(n_classes))


# Model
if args.checkpoint != None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(
        args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint,
                            map_location=lambda storage, loc: storage)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    optimizer = checkpoint['optimizer']
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=5e-4)
else:
    print('==> Building model..')
    if args.pretrained:
        # resnet18, resnet34, resnet50, resnet101, resnet152
        net = ResNet_pretrained("resnet101", 10)
        optimizer = optim.SGD(net.classifier.parameters(),
                              lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        net = ResNet18()
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Schedule LR


def lr_scheduler(optimizer, epoch, init_lr=args.lr, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# Training


# Training
def train(epoch, net, criterion, optimizer, trainloader, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net, criterion, testloader, device, checkpoint_dir):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'class_to_idx': train_dataset.class_to_idx
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, checkpoint_dir)
        best_acc = acc


# Train loop
for epoch in range(start_epoch, start_epoch+args.epochs):
    optimizer = lr_scheduler(
        optimizer, epoch, lr_decay_epoch=args.lr_decay_epoch)
    train(epoch, net, optimizer, criterion)
    test(epoch, net, criterion)
