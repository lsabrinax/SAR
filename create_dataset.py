import os
import glob
import yaml
import lmdb
import cv2
import pdb
import string 
import scipy.io
import numpy as np
import ast
import xml.etree.ElementTree as ET
import pdb

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    # env = lmdb.open(outputPath, map_size=1099511627776)
    num = 0
    with open(outputPath+'gt.txt', 'w') as txt:
        for i in range(nSamples):
            imagePath = imagePathList[i]
            label = labelList[i]

            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                continue
            try:
                cv2.imread(imagePath)
            except:
                print('%s is not a valid image' % imagePath)
                continue
                num += 1
            txt.write(imagePath+' '+label+'\n')
    # cache = {}
    # cnt = 1
    # for i in range(nSamples):
    #     imagePath = imagePathList[i]
    #     label = labelList[i]
    #
    #     if not os.path.exists(imagePath):
    #         print('%s does not exist' % imagePath)
    #         continue
    #
    #     # if checkValid:
    #     #     if not checkImageIsValid(imageBin):
    #     #         print('%s is not a valid image' % imagePath)
    #     #         continue
    #     try:
    #         cv2.imread(imagePath)
    #     except:
    #         print('%s is not a valid image' % imagePath)
    #         continue
    #     with open(imagePath, 'rb') as f:
    #             imageBin = f.read()
    #
    #     imageKey = b'image-%09d' % cnt
    #     labelKey = b'label-%09d' % cnt
    #     cache[imageKey] = imageBin
    #     cache[labelKey] = label.encode()
    #     if lexiconList:
    #         lexiconKey = 'lexicon-%09d' % cnt
    #         cache[lexiconKey] = ' '.join(lexiconList[i])
    #     if cnt % 1000 == 0:
    #         writeCache(env, cache)
    #         cache = {}
    #         print('Written %d / %d' % (cnt, nSamples))
    #     cnt += 1
    # nSamples = cnt-1
    # cache[b'num-samples'] = str(nSamples).encode()
    # writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    FPath=os.getcwd()
    
    for ij in range(6,21):
        imageList=[]
        labelList=[]
        outputPath=FPath+'/DataDB/train_'+str(ij)+'/'   # separate the training data into different groups
        os.makedirs(outputPath)
        labelfile =open('/home/sabrina/data/text-recognition/SAR_SynthAdd/SynthText_Add/annotationlist/gt_'+str(ij)+'.txt')
        imagePathDir ='/home/sabrina/data/text-recognition/SAR_SynthAdd/SynthText_Add/'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            content=line.split(',', 1)#这里换行符是去掉了的
            imagePath=imagePathDir+'crop_img_'+str(ij)+'/'+content[0]
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(content[1][1:-2])
        labelfile.close()

        labelfile = open('/home/sabrina/data/text-recognition/Synth90K/Synth90K/splitimg/gt/gt_%d.txt' % ij)
        imagePathDir ='/home/sabrina/data/text-recognition/Synth90K/Synth90K/splitimg/'
        i = 0
        while i < 120000:
            line = labelfile.readline()
            if not line:
                break
            content=line.strip().split(',', 1)
            imagePath=imagePathDir+str(ij)+'/'+content[0]
            imageList.append(imagePath)
            # ll=len(content[0])
            labelList.append(content[1])
            i += 1
        labelfile.close()

        labelfile = open('/home/sabrina/data/text-recognition/SynthText/cropimg/gt_'+str(ij)+'.txt')
        # labelfile = open('../Dataset/SynthText_org/annotationlist/annotation_train_'+str(ij+99)+'.txt')
        imagePathDir ='/home/sabrina/data/text-recognition/SynthText/cropimg/'
        i = 0
        while i < 120000:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',', 1)
            imagePath=imagePathDir+'/'+str(ij)+'/'+content[0]
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(content[1][:-1])
            i += 1
        labelfile.close()

        # labelfile = open('../Dataset/Max_Syn_90kDICT32px/annotationlist/annotation_train_'+str(ij)+'.txt')
        # imagePathDir ='../Dataset/Max_Syn_90kDICT32px'
        # while 1:
        #     line = labelfile.readline()
        #     if not line:
        #         break
        #     # print(line)
        #     filepath_t=line.split(' ')
        #     filepath=filepath_t[0][1:]
            
        #     labelp=filepath.split('_')
        #     labelList.append(labelp[1])
        #     imagePath=imagePathDir+filepath
        #     imageList.append(imagePath)
        # labelfile.close()

        # labelfile = open('../Dataset/Max_Syn_90kDICT32px/annotationlist/annotation_train_'+str(ij+90)+'.txt')
        # imagePathDir ='../Dataset/Max_Syn_90kDICT32px'
        # while 1:
        #     line = labelfile.readline()
        #     if not line:
        #         break
        #     # print(line)
        #     filepath_t=line.split(' ')
        #     filepath=filepath_t[0][1:]
            
        #     labelp=filepath.split('_')
        #     # if len(labelp)>2:
        #     labelList.append(labelp[1])
        #     imagePath=imagePathDir+filepath
        #     imageList.append(imagePath)
        # labelfile.close()

        labelfile = open('/home/sabrina/data/text-recognition-benchmark/coco/coco-text/train2014/train/gt.txt')
        # labelfile = open('../Dataset/COCO_WordRecognition/train_words_gt.txt')
        imagePathDir ='/home/sabrina/data/text-recognition-benchmark/coco/coco-text/train2014/train/'
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',', 1)
            imagePath=imagePathDir+content[0]
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(content[1][1:-2])
            # try:
                
            # except:
            #     print(line)
            #     print(content)
            #     os._exit()
        labelfile.close()

        imagePathDir='/home/sabrina/data/text-recognition-benchmark/IC15/ch4_training_word_images_gt/split/'
        labelfile =open('/home/sabrina/data/text-recognition-benchmark/IC15/ch4_training_word_images_gt/split/gt.txt')
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',', 1)
            imagePath=imagePathDir+content[0]
            imageList.append(imagePath)
            labelList.append(content[1][1:-2])
            

        labelfile.close()

        imagePathDir='/home/sabrina/data/text-recognition-benchmark/IC13//Challenge2_Training_Task3_Images_GT/'
        labelfile =open('/home/sabrina/data/text-recognition-benchmark/IC13//Challenge2_Training_Task3_Images_GT/gt.txt')
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',', 1)
            imagePath=imagePathDir+content[0]
            imageList.append(imagePath)
            labelList.append(content[1][2:-2])
        labelfile.close()

        imagePathDir='/home/sabrina/data/text-recognition-benchmark/IIIT5K/split'
        labelfile =open('/home/sabrina/data/text-recognition-benchmark/IIIT5K/split/gt.txt')
        while 1:
            line = labelfile.readline()
            if not line:
                break
            # print(line)
            content=line.split(',', 1)
            imagePath=imagePathDir+content[0]
            imageList.append(imagePath)
            ll=len(content[0])
            labelList.append(content[1][1:-2])
        labelfile.close()

        # labelfile =open('/home/sabrina/data/text-recognition-benchmark/coco/coco-text/train2014/val/gt.txt')
        # imagePathDir='/home/sabrina/data/text-recognition-benchmark/coco/coco-text/train2014/val/'
        # while 1:
        #     line = labelfile.readline()
        #     if not line:
        #         break
        #     # print(line)
        #     content=line.split(',', 1)
        #     imagePath=imagePathDir+content[0]
        #     imageList.append(imagePath)
        #     ll=len(content[0])
        #     labelList.append(content[1][1:-2])
        # labelfile.close()

        createDataset(outputPath, imageList, labelList, lexiconList=None, checkValid=True)
