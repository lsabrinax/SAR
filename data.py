import os
import scipy.io as scio
from PIL import Image
import cv2
import numpy

gt_dir = '/home/sabrina/data/text-recognition/SynthText/gt.mat'
img_dir = '/home/sabrina/data/text-recognition/SynthText/'

gt = scio.loadmat(gt_dir)

def get_text(text):
    texts = []
    for t in text:
        texts.extend(t.split())
    return texts

num = 0 #记录裁剪的总的图片数
for ix, imname in enumerate(gt['imnames'][0]):
    # img = Image.open(img_dir + imname[0])
    img = cv2.imread(img_dir + imname[0])
    txts = get_text(gt['txt'][0, ix])
    wordBB = gt['wordBB'][0, ix]
    if len(txts) == 1:
        wordBB = wordBB[:,:,numpy.newaxis]
    assert len(txts) == wordBB.shape[2], "txt length is not compatible with wordbb length"
    for i, t in enumerate(txts):
        wordbb = wordBB[:,:,i]
        x, y , w, h = cv2.boundingRect(wordbb.T)
        try:
            # crop_img = img.crop((x, y, x+w, y+h))
            crop_img = img[y:y+h,x:x+w]
        except:
            print('crop img failed!')
            continue
        cropimg_dir = img_dir + 'cropimg/%d'% ((num % 20) + 1)
        if not os.path.exists(cropimg_dir):
            os.makedirs(cropimg_dir)
        imgname = 'word_%d.jpg' % ((num // 20) + 1)
        # crop_img.save(os.path.join(cropimg_dir, imgname))
        cv2.imwrite(os.path.join(cropimg_dir, imgname), crop_img)
        with open(img_dir + 'cropimg/gt_%d.txt'%((num % 20) + 1), 'a') as f:
            f.write(imgname + ',' + t +'\n')
        num += 1
        print(imname, cropimg_dir, imgname, t, 'done!')