#encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import os

class lmdbDataset(Dataset):
    """docstring for lmdbDataset"""
    def __init__(self, root=None, transform=None, target_transform=None):
        #super(lmdbDataset, self).__init__()
        
        # self.env = lmdb.open(
        #     root,
        #     max_readers=1,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False)
        #
        # if not self.env:
        #     print('cannot creat lmdb from %s' % root)
        #     sys.exit(0)
        #
        # with self.env.begin(write=False) as txn:
        #     nSamples = int(txn.get('num-samples'.encode()).decode())
        #     self.nSamples = nSamples
        with open(os.path.join(root, 'gt.txt'), 'r') as txt:
            self.imgs = txt.readlines()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        line = self.imgs[index]
        content = line.strip().split(' ', 1)
        if len(content) != 2:
            print('line is fomat is wrong!')
            return self[index+1]
        imgpath = content[0]
        label = content[1]
        try:
            img = Image.open(imgpath)
        except:
            print('img has cropted!')
            return self[index+1]
        # index += 1
        # with self.env.begin(write=False) as txn:
        #     img_key = 'image-%09d' % index
        #     imgbuf = txn.get(img_key.encode())
        #     # print(imgbuf)
        #
        #     buf = six.BytesIO()
        #     buf.write(imgbuf)
        #     buf.seek(0)
        #
        #     try:
        #         img = Image.open(buf).convert('L')
        #     except IOError:
        #         print('Corrupted image for %d' % index)
        #         return self[index+1]
        #
        #     if self.transform is not None:
        #         img = self.transform(img)
        #
        #     label_key = 'label-%09d' % index
        #     label = str(txn.get(label_key.encode()).decode())

        return (img, label)


class TestData(Dataset):
    """docstring for TestData"""
    def __init__(self, valroot, labelfile=None):
        
        self.valroot = valroot
        self.labelfile = labelfile
        imgs = os.listdir(valroot)
        self.imgs = [os.path.join(valroot, img) for img in imgs]
        if labelfile != None:
            images = []
            labels = []
            with open(labelfile, 'r') as f:
                for line in f.readlines():
                    content = line.strip().split(' ', 1)
                    images.append(os.path.join(valroot, content[0]))
                    labels.append(content[1])
            self.imgs = images
            self.labels = labels

    def __getitem__(self, index):
        imgpath = self.imgs[index]
        
        # print(imgpath)
        try:
            pilimg = Image.open(imgpath)
        except:
            print('img has Corrupted!')
            return self[index+1]
        if self.labelfile != None:
            label = self.labels[index]
            return (pilimg, label)
        return (pilimg, imgpath)
    def __len__(self):
        return len(self.imgs)


class resizeNormalize(object):
    """docstring for resizeNormalize"""
    def __init__(self, size, interpolation=Image.BILINEAR):
        # super(resizeNormalize, self).__init__()
        self.size = size
        self.interpolation = interpolation
        self.toTenosr = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTenosr(img)
        # print(img.shape)
        img.sub_(0.5).div_(0.5)#这里可以换一下
        return img

class alignCollate(object):
    """docstring for alignCollate"""
    def __init__(self, imgH=48, maxW=160, keep_ratio=True, min_ratio=1):
        # super(alignCollate, self).__init__()
        self.imgH = imgH
        # self.minW = minW
        self.maxW = maxW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        maxW = self.maxW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgW, imgH * self.min_ratio)
            maxW = min(maxW, imgW)


        transform = resizeNormalize((maxW, imgH))
        images = [transform(image) for image in images]
        for img in images:
            print(img.shape)
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels











        





















