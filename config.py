import argparse
import os
import torch

loss_train = 100
loss_test = 100
acc_test = 0
model_train = 'model_train.ckpt'
model_test = 'model_test.ckpt'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainRoot', type=str, default='./fake_all_frames', help='path to training dataset')
    parser.add_argument('--valRoot', type=str, default='./blink_crop_frames', help='path to testing dataset')
    parser.add_argument('--hidden', type=int, default=64, help='number of hidden layers')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize of a data to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--resume', type=str, default="", help='resume training')
    parser.add_argument('--ckpt', type=str, default="./ckpt", help='path to save model checkpoint')
    parser.add_argument('--seqlen', type=int, default=7, help='sequence length')
    parser.add_argument('--seqmax', type=int, default=13, help='maximum sequence length can handle by gpu(constraint by gpu memory)')


    args = parser.parse_args()
    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)
    
    # 从之前训练的模型中load最优的loss
    train_best = os.path.join(args.ckpt, model_train)
    if os.path.exists(train_best):
        ckpt = torch.load(train_best)
        try:
            loss_train = ckpt['loss']
        except:
            pass

    # test loss
    test_best = os.path.join(args.ckpt, model_test)
    if os.path.exists(test_best):
        ckpt = torch.load(test_best)
        try:
            loss_test = ckpt['loss']
            acc_test = ckpt['acc']
        except:
            pass

    return args
