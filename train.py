import visdom
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import RandomSampler
from torch.autograd import Variable as V
import numpy as np
import torch.nn as nn
import os
import utils
import dataset

from model.attnDecoder import SAR, beam_decode

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True, help='visdom port')
parser.add_argument('--trainRoot', required=True, help='path to train dataset')
parser.add_argument('--valRoot', required=True, help='path to val dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgH', type=int, default=48, help='the height of the input image to network')
parser.add_argument('--maxW', type=int, default=160, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=512, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=20, help='number of epoches to train for')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help='path to pretrained model')
parser.add_argument('--lexicon', required=True, help='path to en.lexicon')#具体到文件
parser.add_argument('--expr_dir', default='expr', help='where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=20, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for Critic')
parser.add_argument('--lr_decay', type=float, default=0.9, help = 'learning rate decay')
parser.add_argument('--lr_decay_every', type=int, default=10000, help='when to decay lr')
parser.add_argument('--alpha', type=float, default=0.9, help='alpha for adam')
parser.add_argument('--beta', type=float, default=0.999, help='beta for adam')
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for smoothing')
parser.add_argument('--min_lr', type=float, default=1e-5, help='the min lr should be')
parser.add_argument('--dropout', type=float, default=0.5, help='probability for dropout')
parser.add_argument('--adam', action='store_true', help='whether to use adam')
parser.add_argument('--adadelta', action='store_false', help='whether to use adadelta')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiment')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')


#print('lllll')
opt = parser.parse_args()
print(opt)
vis = visdom.Visdom(env='SAR', port=opt.port)
# writer = SummaryWriter(log_dir='logs')
if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True#输入数据维度或类型变化不大可以这样设置

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda ")

train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
assert train_dataset

#这里没有处理sampler

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, maxW=opt.maxW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(root=opt.valRoot)

converter = utils.strLabelConverter(opt.lexicon)
nclass = converter.nc
nchannel = 1
criterion = nn.CrossEntropyLoss()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)

# encoder = model.RestNet(nchannel, opt.nh)
# decoder = model.attnDecoder(hidden_size=opt.nh, output_size=nclass, batch_size=opt.batchSize, p=opt.dropout)


sar = SAR(nchannel=nchannel, nhidden=opt.nh, output_size=nclass, p=opt.dropout)
sar.apply(weights_init)

if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    sar.load_state_dict(torch.load(opt.pretrained))
print(sar)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.maxW)
text  = torch.FloatTensor(opt.batchSize, 5, nclass)

if opt.cuda:
    #print('convert to gpu')
    text = text.cuda()
    sar = sar.cuda()
    # sar = torch.nn.DataParallel(sar, device_ids=0)
    #print(image.device)
    image = image.cuda()
    #print(image.device)
    criterion = criterion.cuda()
#print(image.device)
imgae = V(image)
#print(image.device)
text = V(text)

loss_avg = utils.averager()
if opt.adam: 
    optimizer = optim.Adam(sar.parameters(), lr=opt.lr, betas=(opt.alpha, opt.beta), eps=opt.epsilon)
elif opt.adadelta:
    optimizer = optim.Adadelta(sar.parameters())
else:
    optimizer = optim.RMSprop(sar.parameters(), lr=opt.lr)

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    #print(cpu_images)
    batch_size = cpu_images.size(0)
    utils.loadData(imgae, cpu_images)
    t, padded = converter.encode(cpu_texts)
    utils.loadData(text, t)
    # padded = V(padded)
    #print(image.device)
    preds, hidden = sar(image, text)
    #print(image.device)
    target = V(padded[:, 1:].contiguous().view(-1))
    if opt.cuda:
        target = target.cuda()
    cost = criterion(preds, target)
    print(preds)
    sar.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def val(net, data_set, criterion, max_iter=100):
    print('Start Val')
    for p in net.parameters():
        p.requires_grad = False
    net.eval()

    data_loader = torch.utils.data.DataLoader(data_set,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, maxW=opt.maxW, keep_ratio=opt.keep_ratio))
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    nsample = 0
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        nsample += batch_size
        utils.loadData(image, cpu_images)
        t, padded = converter.encode(cpu_texts)#padded:inttensor
        utils.loadData(text, t)
        hidden_state, feature_map = net.encoder(image)
        decoder_patch = beam_decode(net.decoder, converter, hidden_state, opt, feature_map)
        #print('decoded_patch is', decoder_patch)
        pred_texts = converter.decode(decoder_patch)
        preds, hidden = sar(image, text)
        padd_target = V(padded[:, 1:].contiguous().view(-1))
        if opt.cuda:
            padd_target = padd_target.cuda()
        cost = criterion(preds, padd_target)
        loss_avg.add(cost)
        for pred, target in zip(pred_texts, cpu_texts):
            # print('pred: %-20s, gt: %-20s' % (pred, target))
            if pred == target:
                n_correct += 1
    accuracy = n_correct / float(nsample)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))
    return accuracy, loss_avg.val()

for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in sar.parameters():
            p.requires_grad = True

        sar.train()
        cost = trainBatch(sar, criterion, optimizer)
        vis.line(X=torch.Tenosr([i]), Y=cost, win='train_loss', update='append' if i > 0 else None, opts={'title': 'train_loss'})
        # writer.add_scalar('train/cost', cost, i)
        loss_avg.add(cost)
        i += 1

        if i % opt.lr_decay_every == 0:
            print('now lr is %f' % opt.lr)
            if opt.lr > opt.min_lr:

                opt.lr = opt.lr * opt.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt.lr
                print('lr is decay by a factor %f, now is %f' %(opt.lr_decay, opt.lr))


        if i % opt.displayInterval == 0:
            vis.text("[{}/{}][{}/{}] loss: {}<br>".format(epoch, opt.epoch, i, len(train_loader), loss_avg.val()), win='text', \
                update='append' if i != opt.displayInterval else None, opts={'title': 'display_message'})
            print('[%d/%d][%d/%d] loss: %f' %
                    (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.saveInterval == 0:
            torch.save(sar.state_dict(), '{0}/netSAR_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))


        if i % opt.valInterval == 0:
            accu, loss = val(sar, test_dataset, criterion)
            vis.text("Test loss: {}, accuracy: {}".format(loss, accu), win='val_text', update='append' if i != opt.valInterval else None, opts={'title': 'val_message'})
            vis.line(X=torch.Tensor([i]), Y=loss, win='val_loss', update='append' if i != opt.valInterval else None, opts={'title': 'val_loss'})#visdom是否可以处理Variable
            
            # writer.add_scalar('val/accu', accu, i)
            # writer.add_scalar('val/loss', loss, i)






