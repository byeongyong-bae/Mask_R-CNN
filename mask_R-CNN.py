#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN을 이용한 X-ray 폐렴 detection

# In[1]:


import numpy as np
import pandas as pd
import os 
import sys
import random
import math
import cv2
import json
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# X-ray와 같은 의학용 이미지를 다루기 위한 패키지
import pydicom

# deep learning의 image data augmentation을 위한 패키지
from imgaug import augmenters as iaa


# In[ ]:


# Mask RCNN 설치
get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')


# In[2]:


# Mask R-CNN import
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[3]:


# train, test data dir 설정
data_dir = '/data'
train_dicom_dir = os.path.join(data_dir, 'train_images')
test_dicom_dir = os.path.join(data_dir, 'test_images')


# In[4]:


# 폴더 내 dicom image들을 list로 얻는다.
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir + '/' + '*.dcm')
    return list(set(dicom_fps))


# In[5]:


# 위에 설정한 get_dicom_fps를 통해 image list를 얻는다.
# image_annotations을 통해 image의 주석을 key로 하는 사전을 얻는다.
def parse_dataset(dicom_dif, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp : [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + ' .dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations


# In[6]:


# 실행 시간을 줄이기 위해 매개변수를 다음과 같이 설정
# 기본 config 값을 다음과 같이 대체
class DetectorConfig(Config):
    # 인식 이름을 '폐렴' 으로 설정
    NAME = 'pneumonia'
    
    # 1개의 GPU에 8개의 image를 train 시킨다.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    # 논문에서도 설정한 resnet50 network로 설정
    BACKBONE = 'resnet50'
    
    # class 중 1개는 pnueumonia class이고, 다른 1개는 pnueumonia가 아닌 background
    NUM_CLASSES = 2
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
    # region proposal network의 anchor(지점)의 scale을 지정
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    
    # classifier와 mask의 heads에 들어갈 ROI의 image 개수
    TRAIN_ROIS_PER_IMAGE = 32
    
    # 1개의 image에서 사용할 ground truth instances 최대 개수
    MAX_GT_INSTANCES = 3
    
    # 최종 detection 최대 개수
    DETECTION_MAX_INSTANCES = 3
    
    # detect된 instance를 허용할 최소 확률
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # detection을 위한 임계 값
    DETECTION_NMS_THRESHOLD = 0.1
    
    # epochs
    STEPS_PER_EPOCH = 100
    
config = DetectorConfig()
config.display()


# In[ ]:


# pneumonia detection을 위한 dataset 
class DetectorDataset(utils.Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # class 추가
        self.add_class('pneumonia', 1, 'Lung Opacity')
        
        # image 추가
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
        
    # image의 참고사항들
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
    
    # image 부르기
    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # 회색이면 RGB값으로 대체
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image
    
    # 
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

