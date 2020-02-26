import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data
import model
import config
import os
import sys
from utils import AverageMeter, default_checkpoint, save_model

cfg = config.parse_args()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
nh = cfg.hidden
epochs = cfg.epochs
batch_size = cfg.batchsize
learning_rate = cfg.lr
pretrained = cfg.resume
seqlen = cfg.seqlen
start_epoch = 0

# MNIST dataset
train_dataset = data.BlinkDataset(cfg.trainRoot, seq_len=seqlen)

test_dataset = data.BlinkDataset(cfg.trainRoot, seq_len=seqlen+2)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

resnetLSTM = model.ResnetLSTM(nh=nh).to(device)
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(resnetLSTM.parameters(), lr=learning_rate)

if pretrained:
    ckpt = default_checkpoint(torch.load(pretrained))
    start_epoch = ckpt['epoch'] if ckpt['epoch'] else 0
    resnetLSTM.load_state_dict(ckpt['ckpt'])
    if ckpt['optimizer']:
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Optimizer loaded.')
    print('Model loaded from ', pretrained)

# Test the model
def test(model, test_loader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.data > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        acc = correct / total
        if acc > config.acc_test:
            save_model(model, os.path.join(cfg.ckpt, config.model_test), acc=acc)
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * acc))


# Train the model
def train():
    total_step = len(train_loader)
    for epoch in range(start_epoch, epochs):
        train_loss_avg = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.float()
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = resnetLSTM(images)
            loss = criterion(outputs, labels)
            train_loss_avg.update(loss.item())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, epochs, i+1, total_step, loss.item()))

        print('End of epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, train_loss_avg.avg))
        if train_loss_avg.avg < config.loss_train:
            save_model(resnetLSTM, os.path.join(cfg.ckpt, config.model_train),
                        epoch=epoch, loss=train_loss_avg.avg,
                        optimizer=optimizer)
            config.loss_train = train_loss_avg.avg

        if epoch % 5 == 0:
            test(resnetLSTM, test_loader)
            resnetLSTM.train()


if __name__ == '__main__':
    #train()
    test(resnetLSTM, test_loader)