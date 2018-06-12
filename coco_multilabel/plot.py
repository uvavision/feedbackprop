import argparse
import os
import numpy as np
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expDir', type = str)
    args = parser.parse_args()

    trainP = os.path.join(args.expDir, 'train.csv')
    trainData = np.loadtxt(trainP, delimiter=',').reshape(-1, 3)
    testP = os.path.join(args.expDir, 'test.csv')
    testData = np.loadtxt(testP, delimiter=',').reshape(-1, 3)
    accuracyP = os.path.join(args.expDir, 'accuracy.csv')
    accuracyData = np.loadtxt(accuracyP, delimiter=',',
                              dtype={'names': ('epoch', 'split', 'meanAP'),
                                     'formats': ('i4', 'S5', 'f4')})

    trainI, trainLoss, trainErr = np.split(trainData, [1, 2], axis=1)
    trainI, trainLoss, trainErr = [x.ravel() for x in
                                   (trainI, trainLoss, trainErr)]

    trainI_ = np.unique(trainI)
    trainLoss_ = np.zeros_like(trainI_)
    trainErr_ = np.zeros_like(trainLoss_)
    for (idx, epoch) in enumerate(trainI_):
        ids = np.nonzero(trainI == epoch)[0]
        trainLoss_[idx] = trainLoss[ids].mean()
        trainErr_[idx] = trainErr[ids].mean()

    testI, testLoss, testErr = np.split(testData, [1, 2], axis = 1)
    epochI = np.array([x[0] for x in accuracyData])
    split = np.array([x[1] for x in accuracyData])
    meanAP = np.array([x[2] for x in accuracyData])

    fig, ax = plt.subplots(1, 1, figsize = (6, 5))
    # plt.plot(trainI, trainLoss, label='Train')
    plt.plot(trainI_, trainLoss_, label = 'Train')
    plt.plot(testI, testLoss, label = 'Test')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(min(trainI_), max(trainI_) + 1, 1.0))
    plt.ylabel('Loss')
    plt.legend()
    loss_fname = os.path.join(args.expDir, 'loss.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

    fig, ax = plt.subplots(1, 1, figsize = (6, 5))
    # plt.plot(trainI, trainErr, label='Train')
    plt.plot(trainI_, trainErr_, label = 'Train')
    plt.plot(testI, testErr, label = 'Test')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(min(trainI_), max(trainI_) + 1, 1.0))
    plt.ylabel('Error')
    plt.legend()
    err_fname = os.path.join(args.expDir, 'error.png')
    plt.savefig(err_fname)
    print('Created {}'.format(err_fname))

    train_ids = np.nonzero([x == 'train' for x in split])[0].astype(int)
    test_ids = np.nonzero([x == 'test' for x in split])[0].astype(int)
    trainEpochI = epochI[train_ids]
    testEpochI = epochI[test_ids]
    trainMeanAP = meanAP[train_ids]
    testMeanAP = meanAP[test_ids]

    fig, ax = plt.subplots(1, 1, figsize = (6, 5))
    # plt.plot(trainI, trainErr, label='Train')
    plt.plot(trainEpochI, trainMeanAP, label = 'Train')
    plt.plot(testEpochI, testMeanAP, label = 'Test')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(min(trainEpochI), max(trainEpochI) + 1, 1.0))
    plt.ylabel('meanAP')
    plt.legend()
    accuracy_fname = os.path.join(args.expDir, 'accuracy.png')
    plt.savefig(accuracy_fname)
    print('Created {}'.format(accuracy_fname))


    loss_err_fname = os.path.join(args.expDir, 'loss-error-accuracy.png')
    os.system('convert +append {} {} {} {}'.format(loss_fname, err_fname, accuracy_fname, loss_err_fname))
    print('Created {}'.format(loss_err_fname))

if __name__ == '__main__':
    main()
