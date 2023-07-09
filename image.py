import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if True:
        orig_img_h, orig_img_w = img.size[0], img.size[1]
        if (img.size[0] > img.size[1]):
            min_size = img.size[1]
        else:
            min_size = img.size[0]
        crop_size = (min_size,min_size) # (width, height)
        dx = int(random.random()*(img.size[0]-crop_size[0])*1.)
        dy = int(random.random()*(img.size[1]-crop_size[1])*1.)
        if dx <= 0:
            dx = 0
        if dy <= 0:
            dy = 0
        img = img.crop((dx,dy,int(crop_size[0])+dx,int(crop_size[1])+dy))
        target = target[dy:int(crop_size[1])+dy,dx:int(crop_size[0])+dx]
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        img_np = np.array(img)
        img_resized = cv2.resize(img_np, (640, 640), interpolation = cv2.INTER_CUBIC)        
        img = Image.fromarray(img_resized)
        
    
#    # w1, h1
#    w1, h1 = target.shape[1], target.shape[0]
#    # w2, h2
#    if (w1 % 2 == 0):
#        w2 = int(w1/2)
#    else:
#        w2 = int(int(w1/2)+1)
#    if (h1 % 2 == 0):
#        h2 = int(h1/2)
#    else:
#        h2 = int(int(h1/2)+1)
#    # w3, h3
#    if (w2 % 2 == 0):
#        w3 = int(w2/2)
#    else:
#        w3 = int(int(w2/2)+1)
#    if (h2 % 2 == 0):
#        h3 = int(h2/2)
#    else:
#        h3 = int(int(h2/2)+1)
#    # w4, h4
#    if (w3 % 2 == 0):
#        w4 = int(w3/2)
#    else:
#        w4 = int(int(w3/2)+1)
#    if (h3 % 2 == 0):
#        h4 = int(h3/2)
#    else:
#        h4 = int(int(h3/2)+1)

     
    target = cv2.resize(target,(int(640/8),int(640/8)),interpolation = cv2.INTER_CUBIC)*(64*min_size*min_size)/(640*640)
    #target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    # target = cv2.resize(target,(w4,h4),interpolation = cv2.INTER_CUBIC)*(w1/w4)*(h1/h4)

    # print((w1/w4)*(h1/h4))
    
    
    return img,target
