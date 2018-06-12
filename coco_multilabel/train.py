import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score
import os, sys, math, pdb, random, string, string, shutil, time, pickle
import numpy as np
import argparse

import data_loader
import model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type = int, default = 64)
    parser.add_argument('--nEpochs', type = int, default = 60)
    parser.add_argument('--startEpoch', type = int, default = 1)
    parser.add_argument('--learningRate', type = float, default = 1e-3)
    parser.add_argument('--logDir', required = True)
    parser.add_argument('--resume', action = 'store_true', default = False, help = 'whether to resume from logDir if existent')
    parser.add_argument('--finetune', action = 'store_true', default = False)

    parser.add_argument('--imageSize', type = int, default = 256)
    parser.add_argument('--cropSize', type = int, default = 224)
    parser.add_argument('--seed', type = int, default = 1)

    parser.add_argument('--annDir', default = 'COCO annotation path')
    parser.add_argument('--imageDir', default = 'COCO image path')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.logDir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.logDir))
        return
    if not os.path.exists(args.logDir): os.makedirs(args.logDir)

    # Log training parameters
    with open(os.path.join(args.logDir, "arguments.txt"), "a") as f:
        f.write(str(args))

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]

    normTransform = transforms.Normalize(normMean, normStd)
    trainTransform = transforms.Compose([
        transforms.Resize(args.imageSize),
        transforms.RandomCrop(args.cropSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.Resize(args.imageSize),
        transforms.CenterCrop(args.cropSize),
        transforms.ToTensor(),
        normTransform
    ])

    # Load data for training.
    trainData, valData = data_loader.loadData(args, trainTransform, testTransform)

    # Prepare data loaders.
    trainLoader = DataLoader(trainData, batch_size = args.batchSize,
                             shuffle = True, pin_memory = True, num_workers = 6)
    testLoader = DataLoader(valData, batch_size = args.batchSize, shuffle = False,
                            pin_memory = False, num_workers = 4)

    # Prepare model.
    print('\nPreparing model...')
    net = model.CocoMultilabelModel(args, trainData.numCategories())
    net = nn.DataParallel(net).cuda()

    # print('model summary:...')
    # print(net)

    best_error = 10000
    if args.resume:
        trainF = open(os.path.join(args.logDir, 'train.csv'), 'a')
        accuracyF = open(os.path.join(args.logDir, 'accuracy.csv'), 'a')
        testF = open(os.path.join(args.logDir, 'test.csv'), 'a')
        if os.path.isfile(os.path.join(args.logDir, 'checkpoint.pth.tar')):
            checkpoint = torch.load(os.path.join(args.logDir, 'checkpoint.pth.tar'))
            args.startEpoch = checkpoint['epoch']
            best_error = checkpoint['error']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        trainF = open(os.path.join(args.logDir, 'train.csv'), 'w')
        accuracyF = open(os.path.join(args.logDir, 'accuracy.csv'), 'w')
        testF = open(os.path.join(args.logDir, 'test.csv'), 'w')


    labelWeights = trainData.getLabelWeights()
    criterion = nn.BCELoss(weight = torch.FloatTensor(labelWeights).cuda(), size_average = True)
    cudnn.benchmark = True

    def get_trainable_parameters(model):
        for param in model.parameters():
            if param.requires_grad:
                yield param

    optimizer = optim.SGD(get_trainable_parameters(net), lr = args.learningRate,
                          momentum = 0.9, weight_decay = 1e-4)

    for epoch in range(args.startEpoch, args.nEpochs + 1):
        train(args, epoch, net, criterion, trainLoader, optimizer, trainF, accuracyF)
        error = test(args, epoch, net, criterion, testLoader, optimizer, testF, accuracyF)
        is_best = error < best_error
        best_error = min(error, best_error)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'error': error
        }, is_best, os.path.join(args.logDir, 'checkpoint.pth.tar'))
        os.system('python plot.py {} &'.format(args.logDir))

    trainF.close()
    testF.close()
    accuracyF.close()

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.logDir, 'model_best.pth.tar'))

def train(args, epoch, net, criterion, trainLoader, optimizer, trainF, accuracyF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_logger = AverageMeter()
    error_logger = AverageMeter()

    results = list()

    end = time.time()
    for batch_idx, batch_content in enumerate(trainLoader):
        # if batch_idx == 100: break
        # measure data loading time

        data, target, imageIds = batch_content

        data_time.update(time.time() - end)

        # prepare inputs.
        data, target = data.cuda(), target.cuda()

        # Forward pass.
        optimizer.zero_grad()
        data, target = Variable(data), Variable(target)
        output = net(data)
        loss = criterion(output, target)
        loss_logger.update(loss.data[0])
        loss.backward()
        optimizer.step()

        # Compute time elapsed on batch.
        batch_time.update(time.time() - end)
        end = time.time()

        results.append((imageIds, output.data.cpu(), target.data.cpu()))

        # Compute logging information.
        nProcessed += len(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = torch.gather(target.data, 1, pred).cpu().sum()
        err = 100.* (1 - correct * 1. / len(data))
        error_logger.update(err)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {loss.val:.6f} ({loss.avg:.6f})'
              '\tError: {err.val:.2f} ({err.avg:.2f})\t'
              'Time {batch_time.val:.3f} -- '
              'Data {data_time.val:.3f}\t'.format(
            epoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss = loss_logger, err = error_logger, batch_time = batch_time, data_time = data_time))

        trainF.write('{},{},{}\n'.format(epoch, loss.data[0], err))
        trainF.flush()

    imageIds = torch.cat([entry[0] for entry in results], 0)
    predictions = torch.cat([entry[1] for entry in results], 0)
    targets = torch.cat([entry[2] for entry in results], 0)

    meanAP = average_precision_score(targets.numpy(), predictions.numpy())
    accuracyF.write('{},{},{}\n'.format(epoch, 'train', meanAP))
    accuracyF.flush()

def test(args, epoch, net, criterion, testLoader, optimizer, testF, accuracyF):
    net.eval()
    test_loss = 0
    correct = 0
    results = list()
    for batch_idx, batch_content in enumerate(testLoader):
        # if batch_idx == 50: break
        # Prepare inputs.
        data, target, imageIds = batch_content
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile = True), Variable(target)

        # Forward pass.
        output = net(data)
        loss = criterion(output, target)
        batch_loss = loss.data[0]

        # Logging information.
        results.append((imageIds, output.data.cpu(), target.data.cpu()))
        test_loss += batch_loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += torch.gather(target.data, 1, pred).cpu().sum()

    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*(1 - correct / nTotal)
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct * 1., nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    imageIds = torch.cat([entry[0] for entry in results], 0)
    predictions = torch.cat([entry[1] for entry in results], 0)
    targets = torch.cat([entry[2] for entry in results], 0)

    meanAP = average_precision_score(targets.numpy(), predictions.numpy())
    accuracyF.write('{},{},{}\n'.format(epoch, 'test', meanAP))
    accuracyF.flush()

    return (1 - meanAP)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=='__main__':
    main()
