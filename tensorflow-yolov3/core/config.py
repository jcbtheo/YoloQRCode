#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "./data/classes/coco.names"
__C.YOLO.ANCHORS                = "./data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "../combined5mega.txt"
__C.TRAIN.BATCH_SIZE            = 5
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 0
__C.TRAIN.SECOND_STAGE_EPOCHS   = 35
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/combined5.1/testloss=1.5611.ckpt-20"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "../combined5megatest.txt"
__C.TEST.BATCH_SIZE             = 3
__C.TEST.INPUT_SIZE             = 416
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_test_loss=3.0350.ckpt-10"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.4
__C.TEST.IOU_THRESHOLD          = 0.45






