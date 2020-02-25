import resnet
import torch
import torch.nn as nn

def resnet18_feat(model):
    func = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, 
                          model.layer2, model.layer3, model.layer4)
    return func


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.reshape(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class ResnetLSTM(nn.Module):
    def __init__(self, nh, pretrained=True):
        super().__init__()
        self.root_model = resnet.resnet18(pretrained)
        self.feat = resnet18_feat(self.root_model)
        self.activation = nn.MaxPool2d(kernel_size=2)
        self.rnn = nn.Sequential(
                            BidirectionalLSTM(512, nh, nh), # 512是resnet18的feat的深度
                            BidirectionalLSTM(nh, nh, 1),
                            nn.Sigmoid())
        
    def forward(self, x):
        if len(x.size()) == 5:
            b, l, c, h, w = x.size()
            x = x.view((b*l, c, h, w))
        else:
            b = 1
        feat = self.feat(x)
        feat = self.activation(feat)
        #print('raw feat:', feat.size())
        feat = feat.squeeze(-1).squeeze(-1) # (b×c, w)
        if b == 1:
            feat = feat.unsqueeze(0)
        else:
            feat = feat.view((b, l, -1))
        #feat = feat.permute(0, 0, 1) # (w, c, b) (input_dim, batch, seq_len)
        #print('rnn input:', feat.size())
        
        classification = self.rnn(feat)
        #print('class:', classification.size())
        return classification[:, -1, 0]


if __name__ == '__main__':
    import data
    dataset = data.BlinkDataset('./fake_all_frames')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2)

    resnet18 = resnet.resnet18(pretrained=True)
    resnet18_ = resnet18_feat(resnet18)

    for item in dataloader:
        r, _ = item
        print(r.size())
        #print(resnet18_(r).size())

        model = ResnetLSTM(nh=64)
        print(model(r).size())

        break