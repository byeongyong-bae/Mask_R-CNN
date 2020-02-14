# Mask R-CNN을 이용한 X-ray 폐렴 detection

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

# Mask RCNN 설치
get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')

# Mask R-CNN import
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# train, test data dir 설정
data_dir = '/data'
train_dicom_dir = os.path.join(data_dir, 'train_images')
test_dicom_dir = os.path.join(data_dir, 'test_images')

# 폴더 내 dicom image들을 list로 얻는다.
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir + '/' + '*.dcm')
    return list(set(dicom_fps))

# 위에 설정한 get_dicom_fps를 통해 image list를 얻는다.
# image_annotations을 통해 image의 주석을 key로 하는 사전을 얻는다.
def parse_dataset(dicom_dif, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp : [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + ' .dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations

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
    
    # mask 생성
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

 # train label data set
anns = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))

# image와  label을 더한 train data set 설정
image_fps , image_annotations = parse_dataset(train_dicom_dir, anns = anns)

# dicom image 부르기
ds = pydicom.read_file(image_fps[0])

# image arrary
image = ds.pixel_array

# dicom imager가 1024 x 1024
ORIG_SIZE = 1024

# 실행시간과 GPU 환경 설정에 맞춰서 일부만 수행
image_fps_list = list(image_fps[:1000])

# train와 validatoin 분리
sorted(image_fps_list)
random.seed(1030)
random.shuffle(image_fps_list)
validation_split = 0.1
split_index = int((1 - validation_split) * len(image_fps_list))
image_fps_train = image_fps_list[:split_index]
image_fps_val = image_fps_list[split_index:]

# Detector Datatset에 맞춰서 train data set 준비
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

# 샘플을 확인
test_fp = random.choice(image_fps_train)
image_annotations[test_fp]

# Detector Dataset에 맞춰서 validatoin data set 준비
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()

# 샘플로 bounding box와 함께 image 확인
image_id = random.choice(dataset_train.image_ids)
image_fp = dataset_train.image_reference(image_id)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
print(image.shape)

# 시각화
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap='gray')
plt.axis('off')
print(image_fp)
print(class_ids)

# model 설정 및 결과 및 저장
root_dir = 'data/working'
model = modellib.MaskRCNN(mode = 'training', config = config, model_dir = root_dir)

# image augmentaion finetuning 설정
augmentation = iaa.SomeOf((0, 1), [
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    ),
    iaa.Multiply((0.9, 1.1))
])

# 학습과 finetuning
# 노트북 환경과 시간을 고려해 epoch는 1
NUM_EPOCHS = 1

import warnings 
warnings.filterwarnings("ignore")
model.train(dataset_train, dataset_val, 
            learning_rate = config.LEARNING_RATE, 
            epochs = NUM_EPOCHS, 
            layers = 'all',
            augmentation = augmentation)

# train된 model 선택
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))
    
fps = []
# 미자막 저장위치 가져오기
for d in dir_names: 
    dir_name = os.path.join(model.model_dir, d)
    # 마지막 checkpoint 가져오기
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = sorted(checkpoints)
    
    if not checkpoints :
        print('{}에 파일 없음'.format(dir_name))
    else:
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        fps.append(checkpoint)
        
model_path = sorted(fps)[-1]
print('최종 모델 {}'.format(model_path))

# finetuning된 parameter로 inference model 재설정
# inference config 설정
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# inference model을 재생성
model = modellib.MaskRCNN(mode = 'inference', 
                          config = inference_config,
                          model_dir = root_dir)

# 학습된 가중치를 불러오기
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# class의 색깔 설정
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors

# validatoin data로 predict box와 expected value를 어떻게 비교되는지 validation으로 체크
# ground truth vs validatoin prediction
dataset = dataset_val
fig = plt.figure(figsize = (10, 30))

for i in range(4):
    image_id = random.choice(dataset.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
    print(original_image.shape)
    
    plt.subplot(6, 2, 2*i + 1)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset.class_names, colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
    
    plt.subplot(6, 2, 2*i + 2)
    results = model.detect([original_image])
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], 
                                colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
