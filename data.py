import torch
import torch.nn as nn
from torch.utils.data import Dataset
import PIL.Image as Image
import glob
import os
import pickle
import torchvision
import torchvision.transforms as transforms
import numpy as np

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


def get_transform(size=64, flip=0):
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(size+4),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(flip),
        transforms.ToTensor(),
        normalize
    ])
    return transform_train

class BlinkDataset(Dataset):
    def __init__(self, root, seq_len=7, transform=get_transform()):
        super().__init__()
        self.seq_len = seq_len
        '''
            root 
                - vid1
                    -- 0001.jpg
                    -- 0001.pkl
                    -- 0002.jpg
                    -- 0002.pkl
                       ...
                    -- label.pkl
                - vid2
                  ...
        '''
        self.videos = glob.glob(os.path.join(root, '*'))

        self.label = {}
        for video in self.videos:
            label_file = os.path.join(video, 'label.pkl')
            assert os.path.exists(label_file)
            with open(label_file, 'rb') as f:
                self.label[video] = pickle.load(f)
        
        # label each sequence with a index number
        self.index = {}
        idx = 0
        for label in self.label.keys(): # iterate through all videos
            video_size = len(self.label[label])
            #print(label)
            #print(self.label[label])
            keys = list(self.label[label].keys())
            for i in range(0, video_size-self.seq_len):
                self.index[idx] = os.path.join(label, keys[i])
                idx += 1

        self.len = idx
        self.transform = transform

    def __len__(self):
        return self.len

    def replace_with_index(self, file, index):
        folder, file = file.rsplit('/', 1)
        postfix = file.rsplit('.', 1)[-1]
        return os.path.join(folder, '%04d.%s'%(index, postfix))

    def __getitem__(self, idx):
        assert idx < self.len
        batch = []

        # extract start index from starting point
        seq_label = 0

        starting_file = self.index[idx]
        vid, frame = starting_file.rsplit('/', 1)
        frame_labels = self.label[vid]
        starting_index = int(frame.rsplit('.', 1)[0])
        for i in range(self.seq_len):
            try:
                seq_label += frame_labels['%04d.jpg'%(starting_index+i)]
            except KeyError:
                #print('%04d.pkl missing in %s'%(starting_index+i, vid))
                pass

            frame_file = self.replace_with_index(starting_file, starting_index+i)
            assert os.path.exists(frame_file)
            frame = Image.open(frame_file)
            frame = self.transform(frame)
            frame = frame.unsqueeze(0)
            batch.append(frame)
        batch_ = torch.cat(batch)
        blinked = 1 if (seq_label>0 and seq_label<self.seq_len) else 0
        return batch_, blinked


if __name__ == '__main__':
    dataset = BlinkDataset('./fake_all_frames')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2)
    cnt_freeze, cnt_blinked = 0, 0
    for item in dataloader:
        batch, blinked = item
        print(batch.size(), blinked.size(), len(batch.size()))
        break