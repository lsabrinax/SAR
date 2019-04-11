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
parser.add_argument('--env', default='SAR', help='env to display message')
parser.add_argument('--type', required=True, help='train or val')
parser.add_argument('--gpuid', type=int, required=True, help='which gpu to run')
parser.add_argument('--port', type=int, required=True, help='visdom port')
parser.add_argument('--trainRoot', required=True, help='path to train dataset')
parser.add_argument('--valRoot', required=True, help='path to val dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgH', type=int, default=48, help='the height of the input image to network')
parser.add_argument('--maxW', type=int, default=160, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=512, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=10, help='number of epoches to train for')

parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help='path to pretrained model')
parser.add_argument('--lexicon', default='/home/sabrina/SAR/data/en.lexicon', help='path to en.lexicon')#具体到文件
parser.add_argument('--expr_dir', required=True, help='where to store samples and models')
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
parser.add_argument('--keep_ratio', default=True, help='whether to keep ratio')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiment')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')


opt = parser.parse_args()
print(opt)


if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True#输入数据维度或类型变化不大可以这样设置

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda ")


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


sar = SAR(nchannel=nchannel, nhidden=opt.nh, output_size=nclass, p=opt.dropout, batch_size=opt.batchSize)
sar.apply(weights_init)

if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    sar.load_state_dict(torch.load(opt.pretrained, map_location=lambda storage, loc: storage.cuda(opt.gpuid)))
print(sar)

if opt.cuda:
    sar.cuda(opt.gpuid)
    criterion.cuda(opt.gpuid)

loss_avg = utils.averager()
if opt.adam: 
    optimizer = optim.Adam(sar.parameters(), lr=opt.lr, betas=(opt.alpha, opt.beta), eps=opt.epsilon)
elif opt.adadelta:
    optimizer = optim.Adadelta(sar.parameters())
else:
    optimizer = optim.RMSprop(sar.parameters(), lr=opt.lr)

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
    max_iter = len(data_loader)

    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        nsample += batch_size
        t, padded = converter.encode(cpu_texts)#padded:inttensor
        imgae = V(cpu_images)
        text = V(t)
        padd_target = V(padded[1:, :].contiguous().view(-1))
        if opt.cuda:
            text = text.cuda(opt.gpuid)
            image = imgae.cuda(opt.gpuid)
            padd_target = padd_target.cuda(opt.gpuid)
        
        hidden_state, feature_map = net.encoder(image)
        decoder_patch = beam_decode(net.decoder, converter, hidden_state, opt, feature_map)
        pred_texts = converter.decode(decoder_patch)
        preds, hidden = sar(image, text) 
        cost = criterion(preds, padd_target)
        loss_avg.add(cost)
        for pred, target in zip(pred_texts, cpu_texts):
            print('pred: %-20s, gt: %-20s' % (pred, target))
            if pred == target:
                n_correct += 1
    accuracy = n_correct / float(nsample)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))
    # return accuracy, loss_avg.val()

def test(net):
    print('Start Test!')
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    test_dataset = dataset.TestData(opt.valRoot)
    n_correct = 0
    nsample = 0
    for img in test_dataset:
        w, h = img.size
        transform = dataset.resizeNormalize((opt.imgH, int(w / float(h) * opt.imgH)))
        img = transform(img)
        image = V(img).unsqueeze(0)
        if opt.cuda:
            image = image.cuda(opt.gpuid)
        hidden_state, feature_map = net.encoder(image)
        decoder_patch = beam_decode(net.decoder, converter, hidden_state, opt, feature_map)
        pred_texts = converter.decode(decoder_patch)
        print('pred: %-20s' % pred_texts[0])

def train():
    vis = visdom.Visdom(env=opt.env, port=opt.port)
    sar.train()
    loss_avg.reset()
    ij = 0
    mes = ''
    for i in range(1, 21):
        trainroot = os.path.join(opt.trainRoot, 'train_%d'%i)
        train_dataset = dataset.lmdbDataset(root=trainroot)
        assert train_dataset
        data_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
            collate_fn=dataset.alignCollate(imgH=opt.imgH, maxW=opt.maxW, keep_ratio=opt.keep_ratio))

        for y in range(1, 3):
            iy = 0
            for data, label in data_loader:
                t, padded = converter.encode(label)
                img = V(data)
                txt = V(t)
                target = V(padded[1:, :].contiguous().view(-1))
                if opt.cuda:
                    img = img.cuda(opt.gpuid)
                    txt = txt.cuda(opt.gpuid)
                    target = target.cuda(opt.gpuid)
                preds, hidden = sar(img, txt)
                cost = criterion(preds, target)
                sar.zero_grad()
                cost.backward()
                optimizer.step()
                iy += 1
                ij += 1
                loss_avg.add(cost)
                if ij % 20 == 0:
                    vis.line(X=torch.Tensor([ij]), Y=cost.data.view(-1), win='train_loss', update='append' if ij > 20 else None, opts={'title': 'train_loss'})
                num = (i - 1) * 2 + y
                if ij % opt.displayInterval == 0:
                    
                    mes += "[{}/{}][{}/{}] loss: {}<br>".format(num, 40, iy, len(data_loader), loss_avg.val())
                    loss_avg.reset()
                    vis.text(mes, win='text', opts={'title': 'display_message'})

                if ij % opt.saveInterval == 0:
                     torch.save(sar.state_dict(), '{0}/netSAR_{1}_{2}.pth'.format(opt.expr_dir, num, iy))

                if ij % opt.lr_decay_every == 0:
                    print('now lr is %f' % opt.lr)
                    if opt.lr > opt.min_lr:

                        opt.lr = opt.lr * opt.lr_decay
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = opt.lr
                        print('lr is decay by a factor %f, now is %f' %(opt.lr_decay, opt.lr))
        train_dataset.close()

def train_normal():
    vis = visdom.Visdom(env=opt.env, port=opt.port)
    sar.train()
    loss_avg.reset()
    mes = ''
    train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
            collate_fn=dataset.alignCollate(imgH=opt.imgH, maxW=opt.maxW, keep_ratio=opt.keep_ratio))
    for epoch in range(opt.nepoch):

        for ii, (data, label) in enumerate(data_loader):
            img = V(data)
            t, padded = converter.encode(label)
            txt = V(t)
            target = V(padded[1:, :].contiguous().view(-1))
            if opt.cuda:
                img = img.cuda(opt.gpuid)
                txt = txt.cuda(opt.gpuid)
                target = target.cuda(opt.gpuid)
            try: 
                preds, hidden = sar(img, txt)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('|WARNING: ran out of memory!')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
            cost = criterion(preds, target)
            sar.zero_grad()
            cost.backward()
            optimizer.step()
            loss_avg.add(cost)
            ii += 1
            num = len(data_loader) * epoch + ii
            if ii % 10 == 0:
                vis.line(X=torch.Tensor([ii]), Y=cost.data.view(-1), win='train_loss', update='append' if ii > 10 else None, opts={'title': 'train_loss'})
            
            if num % opt.displayInterval == 0:
                
                mes += "[{}/{}][{}/{}] loss: {}<br>".format(epoch, opt.nepoch, ii, len(data_loader), loss_avg.val())
                loss_avg.reset()
                vis.text(mes, win='text', opts={'title': 'display_message'})

            if num % opt.saveInterval == 0:
                 torch.save(sar.state_dict(), '{0}/netSAR_{1}_{2}.pth'.format(opt.expr_dir, epoch, ii))

            
            if num % opt.lr_decay_every == 0:
                print('now lr is %f' % opt.lr)
                if opt.lr > opt.min_lr:

                    opt.lr = opt.lr * opt.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = opt.lr
                    print('lr is decay by a factor %f, now is %f' %(opt.lr_decay, opt.lr))
        

if __name__ == '__main__':

    if opt.type == 'train':
        train()
    elif opt.type == 'val':
        valRoot = opt.valRoot
        test_dataset = dataset.lmdbDataset(root=valRoot)
        val(sar, test_dataset, criterion)
    elif opt.type == 'test':
        test(sar)
    elif opt.type == 'normal':
        train_normal()


# image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.maxW)
# text  = torch.FloatTensor(opt.batchSize, 5, nclass)

# if opt.cuda:
#     #print('convert to gpu')
#     text = text.cuda(opt.gpuid)
#     sar = sar.cuda(opt.gpuid)
#     # sar = torch.nn.DataParallel(sar, device_ids=0)
#     #print(image.device)
#     image = image.cuda(opt.gpuid)
#     #print(image.device)
#     criterion = criterion.cuda(opt.gpuid)
# #print(image.device)
# imgae = V(image)
# #print(image.device)
# text = V(text)


# val(sar, test_dataset, criterion)
# for epoch in range(opt.nepoch):
#     train_iter = iter(train_loader)
#     i = 0
#     while i < len(train_loader):
#         for p in sar.parameters():
#             p.requires_grad = True

#         sar.train()
#         cost = trainBatch(sar, criterion, optimizer)
#         if i % 10 == 0:
#             vis.line(X=torch.Tensor([i]), Y=cost.data.view(-1), win='train_loss', update='append' if i > 0 else None, opts={'title': 'train_loss'})
#         # writer.add_scalar('train/cost', cost, i)
#         loss_avg.add(cost)
#         i += 1

#         if i % opt.lr_decay_every == 0:
#             print('now lr is %f' % opt.lr)
#             if opt.lr > opt.min_lr:

#                 opt.lr = opt.lr * opt.lr_decay
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = opt.lr
#                 print('lr is decay by a factor %f, now is %f' %(opt.lr_decay, opt.lr))


#         if i % opt.displayInterval == 0:
#             vis.text("[{}/{}][{}/{}] loss: {}<br>".format(epoch, opt.nepoch, i, len(train_loader), loss_avg.val()), win='text', opts={'title': 'display_message'})
#             print('[%d/%d][%d/%d] loss: %f' %
#                     (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
#             loss_avg.reset()

#         if i % opt.saveInterval == 0:
#             torch.save(sar.state_dict(), '{0}/netSAR_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))


#         if i % opt.valInterval == 0:
#             accu, loss = val(sar, test_dataset, criterion)
#             vis.text("Test loss: {}, accuracy: {}".format(loss, accu), win='val_text', opts={'title': 'val_message'})
#             vis.line(X=torch.Tensor([i]), Y=loss.data.view(-1), win='val_loss', update='append' if i != opt.valInterval else None, opts={'title': 'val_loss'})#visdom是否可以处理Variable
            
            # writer.add_scalar('val/accu', accu, i)
            # writer.add_scalar('val/loss', loss, i)

# def trainBatch(net, criterion, optimizer):
#     data = train_iter.next()
#     cpu_images, cpu_texts = data
#     batch_size = cpu_images.size(0)
#     utils.loadData(imgae, cpu_images)
#     t, padded = converter.encode(cpu_texts)
#     utils.loadData(text, t)
#     preds, hidden = sar(image, text)
#     target = V(padded[1:, :].contiguous().view(-1))
#     if opt.cuda:
#         target = target.cuda(opt.gpuid)
#     cost = criterion(preds, target)
#     sar.zero_grad()
#     cost.backward()
#     optimizer.step()
#     return cost
