import os
import scipy.io as scio
from PIL import Image
import cv2
import numpy
import fire
import shutil
import xml.etree.cElementTree as ET
# gt_dir = '/home/sabrina/data/text-recognition/SynthText/gt.mat'
# img_dir = '/home/sabrina/data/text-recognition/SynthText/'

# gt = scio.loadmat(gt_dir)

# def get_text(text):
#     texts = []
#     for t in text:
#         texts.extend(t.split())
#     return texts

# num = 0 #记录裁剪的总的图片数
# for ix, imname in enumerate(gt['imnames'][0]):
#     # img = Image.open(img_dir + imname[0])
#     img = cv2.imread(img_dir + imname[0])
#     txts = get_text(gt['txt'][0, ix])
#     wordBB = gt['wordBB'][0, ix]
#     if len(txts) == 1:
#         wordBB = wordBB[:,:,numpy.newaxis]
#     assert len(txts) == wordBB.shape[2], "txt length is not compatible with wordbb length"
#     for i, t in enumerate(txts):
#         wordbb = wordBB[:,:,i]
#         x, y , w, h = cv2.boundingRect(wordbb.T)
#         try:
#             # crop_img = img.crop((x, y, x+w, y+h))
#             crop_img = img[y:y+h,x:x+w]
#         except:
#             print('crop img failed!')
#             continue
#         cropimg_dir = img_dir + 'cropimg/%d'% ((num % 20) + 1)
#         if not os.path.exists(cropimg_dir):
#             os.makedirs(cropimg_dir)
#         imgname = 'word_%d.jpg' % ((num // 20) + 1)
#         # crop_img.save(os.path.join(cropimg_dir, imgname))
#         cv2.imwrite(os.path.join(cropimg_dir, imgname), crop_img)
#         with open(img_dir + 'cropimg/gt_%d.txt'%((num % 20) + 1), 'a') as f:
#             f.write(imgname + ',' + t +'\n')
#         num += 1
#         print(imname, cropimg_dir, imgname, t, 'done!')

def manage_IC15(gt_dir):

    gt_file = os.path.join(gt_dir, 'gt.txt')
    gt_split_dir = gt_dir + 'split/gt'
    if not os.path.exists(gt_split_dir):
        os.makedirs(gt_split_dir)
    num = 0#记录成功读取的图片
    with open(gt_file, 'r') as f:
        gts = f.readlines()
        for gt in gts:
            split_dir = gt_dir + 'split/%d' % ((num % 20) + 1)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            imgname = gt.strip().split(',', 1)[0]
            label = gt.strip().split(',', 1)[1]
            newname = 'word_%d.png' % ((num // 20) + 1)
            gt_name = 'gt_%d.txt' %((num % 20) + 1)
            try:
                shutil.copy(os.path.join(gt_dir, imgname), os.path.join(split_dir, newname))
                with open(os.path.join(gt_split_dir, gt_name), 'a') as txt:
                    txt.write(newname+','+label+'\n')
            except:
                print('copy failed!')
                continue
            num += 1
            print(imgname, 'done!')

def split_SVT(svt_dir):
    xml_file = svt_dir + 'train.xml'
    trees = ET.parse(xml_file)
    num = 0
    for img in trees.iter(tag='image'):
        imgname = img.find('imageName').text
        for rect in img.iter('taggedRectangle'):
            h = int(rect.get('height'))
            w = int(rect.get('width'))
            x = int(rect.get('x'))
            y = int(rect.get('y'))
            word = rect.find('tag').text
            newname = 'word_%d.png' % ((num // 20) + 1)
            img_dir = svt_dir + 'split/%d/' % ((num % 20 ) + 1)
            if not os.path.join(img_dir):
                os.makedirs(img_dir)
            gt_dir = svt_dir + 'split/gt/'
            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir)
            gt_name = 'gt_%d.txt' % ((num % 20) + 1)

            try:
                inputimg = cv2.imread(svt_dir+imgname)
                print('read img success!')
                cropimg = inputimg[y:y+h,x:x+w]
                print('cropimg success!')
                cv2.imwrite(img_dir+newname, cropimg)
                print('write img success!')
            except:
                print('cop img failed')
                continue
            with oepn(gt_dir+gt_name, 'a') as f:
                f.write(newname+', "'+word+'"\n')
            num += 1
            print(imname,'done!')
if __name__ == '__main__':
    fire.Fire()

