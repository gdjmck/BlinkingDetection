import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

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

#*************************************************************
# model checkpoint
#*************************************************************

def default_checkpoint(ckpt):
    if type(ckpt) == dict:
        return ckpt
    else:
        return {'epoch': 0, 'optimizer': None, 
                'ckpt': ckpt}


# Save the model checkpoint
def save_model(model, path, epoch=None, loss=None, acc=None, optimizer=None):
    torch.save({'ckpt': model.state_dict(),
                'loss': loss, 'acc': acc, 'epoch': epoch,
                'optimizer': optimizer if optimizer is None else optimizer.state_dict()},
                path)


#*************************************************************
# tensorboard
#*************************************************************
class TrainLog(object):
    def __init__(self, save_loc='./ckpt'):
        self.writer = SummaryWriter(os.path.join(save_loc, 'runs'))
        self.step = 0

    def setStep(self, index):
        self.step = index

    def update(self):
        self.step += 1
    
    def addImage(self, img, tag='image'):
        self.writer.add_image(tag, img, self.step)

    def addSeq(self, seq, tag='sequence'):
        grid = torchvision.utils.make_grid(seq)
        self.writer.add_image(tag, grid, self.step)

    def addScalar(self, scalar, tag='loss'):
        self.writer.add_scalar(tag, scalar, self.step)