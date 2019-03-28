import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import numpy as np
from torchvision.transforms import transforms as t
class strLabelConverter(object):
    """docstring for strLabelConverter"""
    def __init__(self, lexicon_file):
        # super(strLabelConverter, self).__init__()

        self.ch2ix = {}
        with open(lexicon_file, 'r') as f:
            
            for ix, line in enumerate(f.readlines()):
                self.ch2ix[line.strip()[0]] = ix

        self.ch2ix['<START>'] = len(self.ch2ix)
        self.ch2ix['<END>'] = len(self.ch2ix)
        self.ch2ix['<PAD>'] = len(self.ch2ix)
        self.ch2ix['<UNK>'] = len(self.ch2ix)
        self.toTensor = t.ToTensor()
        self.ix2ch = {ix: ch for ch, ix in list(self.ch2ix.items())}
        self.nc = len(self.ix2ch)
        
        #调试信息
        print('alphabet length is %d' % self.nc)
        print('alphabet length is {}, 最后一个字符是：{}'.format(len(self.ix2ch), self.ix2ch[len(self.ix2ch) -1]))


    def encode(self, text):
        text = list(text)
        batch_size = len(text)
        text_length = []
        for i in range(batch_size):
            text[i] = ['<START>'] + list(text[i]) + ['<END>']
            text_length.append(len(text[i]))
        
        text = [[self.ch2ix[ch] if ch in self.ch2ix else self.ch2ix['<UNK>'] for ch in line] for line in text]
        #torch.LongTensor(text)
        text_onehot, text = self.oneHot(text_length, text)
        # maxLength = text_length.max()
        # text_onehot = torch.FloatTensor(batch_size, maxLength, self.nc).fill_(0)

        # for i in range(batch_size):

        return text_onehot, text#tensor


    def decode(self, decoder_batch):
        batch_size = len(decoder_batch)
        texts = []
        for i in range(batch_size):
            texts.append(''.join([self.ix2ch[ix]for ix in decoder_batch[i]]))
        return texts


    def oneHot(self, text_length, text):

        batch_size = len(text)
        maxLength = max(text_length)
        padded_text = torch.LongTensor(batch_size, maxLength).fill_(self.ch2ix['<PAD>'])
        text_onehot = torch.FloatTensor(batch_size, maxLength, self.nc).fill_(0)
        for i in range(batch_size):
            length = text_length[i]
            padded_text[i, :length] = torch.Tensor(text[i][:length])
        #print(padded_text)
       # print(text_onehot.shape)
        padded_text = padded_text.view(batch_size, maxLength, 1)
        #print(padded_text.shape)
        text_onehot.scatter_(2, padded_text, 1.0)
        return text_onehot, padded_text.squeeze(-1).permute(1, 0)

def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


class averager(object):
    """docstring for averager"""
    def __init__(self):
        # super(averager, self).__init__()
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res








        
