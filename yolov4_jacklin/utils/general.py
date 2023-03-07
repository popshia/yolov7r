import glob
import math
import os
import random
import shutil
import subprocess
import time
import re
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from sys import platform

import copy
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm.std import trange
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from utils.torch_utils import init_seeds, is_parallel
from utils.box_utils.iou_rotate import iou_rotate_calculate1, iou_rotate_calculate2
from utils.box_utils.nms_rotate import nms_rotate_cpu
from utils.box_utils.rotate_polygon_nms import rotate_gpu_nms
from utils.Rotated_IoU import oriented_iou_loss

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_seeds(seed=seed)


def get_latest_run(search_dir='./runs'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime)


def check_git_status():
    # Suggest 'git pull' if repo is out of date
    if platform in ['linux', 'darwin'] and not os.path.isfile('/.dockerenv'):
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    print('\nAnalyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to generate improved anchors, please wait...' % bpr)
        na = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print('New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    # print('n labels', n)
    # for i in range(n): print('cls count', np.bincount(labels[i][:, 0].astype(np.int), minlength=nc))
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    # print('class_counts', class_counts)
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # print('image_weights', image_weights)
    # print('len image_weights', len(image_weights))
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def fourxy2xywh(fourxy):
    """
    Input:
        shape (n, 8)
        fourxy [[x1,y1,x2,y2,x3,y3,x4,y4], ... , [x1,y1,x2,y2,x3,y3,x4,y4]] 輸入為矩形框的四個順時針角點

    Output:
        shape (n, 4)
        [[x1,y1,w1,h1], ... , [x1,y1,w1,h1]]
    """
    xy = (fourxy[:,[0,1]]+fourxy[:,[4,5]])/2
    wh = ((fourxy[:,[2,4]]-fourxy[:,[0,2]])**2+(fourxy[:,[3,5]]-fourxy[:,[1,3]])**2)**0.5 
    return np.concatenate((xy,wh), axis=1) 


def xywhtheta24xy_new(x):
    """
    numpy ndarray
    dtype float32
    shape (n,5)  n rboxes 5:x, y, w, h, theta
    return (n,8) x1,y1,x2,y2,x3,y3,x4,y4
    """

    
    if isinstance(x, torch.Tensor):
        my_sin = torch.sin
        my_cos = torch.cos
        my_atan2 = torch.atan2
        x = x.double()  # 之前不加沒事，現在不加會突然出現奇怪的overflow error
    elif isinstance(x, np.ndarray):
        my_sin = np.sin
        my_cos = np.cos
        my_atan2 = np.arctan2
        x = x.astype(np.float64)  # 之前不加沒事，現在不加會突然出現奇怪的overflow error
    else:
        print('No supported type')
        exit(0)

    # print('x dtype', x.dtype)
    # print('x type', type(x))
    

    # x[:,4] = np.where(x[:,4] > math.pi, 2*math.pi-x[:,4], x[:,4])
    # hcx = np.cos(x[:,4])*(x[:,3]/2)+x[:,0]

    # a_part = np.sqrt((x[:,3]/2)**2-(hcx-x[:,0])**2)
    # b_part = x[:,1]

    # hcy = np.where(a_part+b_part >= 0, a_part+b_part, b_part-a_part)
    
    # 中心點到任一角點的距離，因為是矩形到四個角點距離相等
    r = (x[:,2]*x[:,2]+x[:,3]*x[:,3])**0.5/2.0
    # print('r', r)
    # 中心點到第一點的向量與中心點到頭中點的向量角度會小於90度所以不用再判斷值是否小於0

    diff_radian = my_atan2((x[:,2]/2),(x[:,3]/2))
    # print('diff_radian', diff_radian)
    # print('my_cos(diff_radian)', my_cos(diff_radian))
    # print('my_sin(diff_radian)', my_sin(diff_radian))
    # print('x[:,2]', x[:,2])
    # print('x[:,3]', x[:,3])
    # print('x[:,2]*x[:,2]', x[:,2]*x[:,2])
    # print('x[:,3]*x[:,3]', x[:,3]*x[:,3])
    # print('x[:,4]', x[:,4])
    # print('r', r)
    # print('x dtype', x.dtype)
    # print('r dtype', r.dtype)

    # if x.dtype not in ['float64', 'float32', 'float16'] or r.dtype not in ['float64', 'float32', 'float16'] or \
    #     x.dtype != torch.float64() or x.dtype != torch.float32() or x.dtype != torch.float16() or \
    #     r.dtype != torch.float64() or r.dtype != torch.float32() or r.dtype != torch.float16():
    #     print('x dtype', x.dtype)
    #     print('r dtype', r.dtype)
    #     exit(0)

    # diff_radian = np.arctan2((x[:,2]/2),(x[:,3]/2))
    # print('diff_radian', diff_radian)
    # 先將矩形平躺到x軸上0度位置，找出四個角點

    p1x, p1y = r*my_cos(diff_radian)+x[:,0], -r*my_sin(diff_radian)+x[:,1]
    p2x, p2y = r*my_cos(diff_radian)+x[:,0], r*my_sin(diff_radian)+x[:,1]
    p3x, p3y = -r*my_cos(diff_radian)+x[:,0], r*my_sin(diff_radian)+x[:,1]
    p4x, p4y = -r*my_cos(diff_radian)+x[:,0], -r*my_sin(diff_radian)+x[:,1]

    # 三角函數是在平面直角座標系上計算角度為逆時針，但直接將角度放在影像座標系時，
    # 因為y軸相反所以會變成順時針，為了使旋轉角正確用2pi減掉原本角度
    r_angle = 2*math.pi-x[:,4]
    # print('r_angle', r_angle)
    #TODO 將公式改為矩陣運算
    # 透過旋轉公式將平躺的矩形旋轉到原來位置，以求得新的四個角點
    x1 = (p1x-x[:,0])*my_cos(r_angle) - (p1y-x[:,1])*my_sin(r_angle) + x[:,0]
    y1 = (p1x-x[:,0])*my_sin(r_angle) + (p1y-x[:,1])*my_cos(r_angle) + x[:,1]
    x2 = (p2x-x[:,0])*my_cos(r_angle) - (p2y-x[:,1])*my_sin(r_angle) + x[:,0]
    y2 = (p2x-x[:,0])*my_sin(r_angle) + (p2y-x[:,1])*my_cos(r_angle) + x[:,1]
    x3 = (p3x-x[:,0])*my_cos(r_angle) - (p3y-x[:,1])*my_sin(r_angle) + x[:,0]
    y3 = (p3x-x[:,0])*my_sin(r_angle) + (p3y-x[:,1])*my_cos(r_angle) + x[:,1]
    x4 = (p4x-x[:,0])*my_cos(r_angle) - (p4y-x[:,1])*my_sin(r_angle) + x[:,0]
    y4 = (p4x-x[:,0])*my_sin(r_angle) + (p4y-x[:,1])*my_cos(r_angle) + x[:,1]

    # First point
    # x1 = r*np.cos(x[:,4]+diff_radian)+x[:,0]
    # y1 = -(r*np.sin(x[:,4]+diff_radian))+x[:,1]

    # # Second point
    # x2 = r*np.cos(x[:,4]-diff_radian)+x[:,0]
    # y2 = -(r*np.sin(x[:,4]-diff_radian))+x[:,1]

    # # Third point
    # x3 = -r*np.cos(x[:,4]+diff_radian)+x[:,0]
    # y3 = r*np.sin(x[:,4]+diff_radian)+x[:,1]

    # # Fourth point
    # x4 = -r*np.cos(x[:,4]-diff_radian)+x[:,0]
    # y4 = r*np.sin(x[:,4]-diff_radian)+x[:,1]

    if isinstance(x, torch.Tensor):
        return torch.stack((x1,y1,x2,y2,x3,y3,x4,y4), dim=0).T
    elif isinstance(x, np.ndarray):
        return np.stack((x1,y1,x2,y2,x3,y3,x4,y4), axis=0).T
    else:
        print('No supported type')
        exit(0)

def r_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (x1y1x2y2x3y3x4y4) from img1_shape to img0_shape

    Input:
        x1y1x2y2x3y3x4y4 before scaled

    Output:
        x1y1x2y2x3y3x4y4 after scaled
    """
    # Rescale coords (x1y1x2y2x3y3x4y4) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    # clip_coords(coords, img0_shape)

    # print('img0_shape', img0_shape)
    # input('r scale coords new pause')
    # clip_rboxes(coords, (0, 0), (img0_shape[1]-1, img0_shape[0]-1))
    
    return coords


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def r_box_iou(box1, box2, useGPU=True):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x, y, w, h, theta) format.
    Arguments:
        box1 (Tensor[N, 5])
        box2 (Tensor[M, 5])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    ious = iou_rotate_calculate1(box1.cpu().numpy(), box2.cpu().numpy(), use_gpu=useGPU, gpu_id=torch.cuda.current_device())
    ious = torch.from_numpy(ious).to(box1.device)

    # numpy to tensor if necessary
    # ious = torch.from_numpy(ious).cuda()

    return ious



def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def count_targets(targets, counts_of_each_cls):
    t_cls_unique = targets[:,1].unique()
    for t_cls_uni in t_cls_unique:
        specify_cls_counts = (targets[:,1]==t_cls_uni).sum().cpu().item()
        t_cls = int(t_cls_uni.cpu().item())
        if t_cls in counts_of_each_cls:
            counts_of_each_cls[t_cls] += specify_cls_counts
        else:
            counts_of_each_cls[t_cls] = specify_cls_counts


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


def compute_loss(p, targets, model, batch_No=-1, num_iter=-1, epoch=-1, theta_format='dhxdhy', loss_terms='hioudhxdhy', num_extra_outputs=0, \
                n_loss_terms=3, tb_writer=None, imgs=None, cv_imgs=None, log_dir='', use_angle_modify_factor=False, counts_of_each_cls={}):  # predictions, targets, model
    """
    default:
        train_mode='dxdy'
        num_extra_outputs=2  (dhx,dhy)
        n_loss_terms=3  (obj loss, cls loss, total loss)

    Input:
        p: [small_forward, medium_forward, large_forward]  eg:small_forward.size=(batch_size, 3種scale anchor框, size1, size2, no)
            small_forward: batch_size, 3小anchor框, size1=network_input_size_h / 8, size2=network_input_size_w / 8, no=(x,y,w,h,conf,cls1,...,clsn,extra_1,...,extra_n)
            medium_forward: batch_size, 3中anchor框, size1=network_input_size_h / 16, size2=network_input_size_w / 16, no=(x,y,w,h,conf,cls1,...,clsn,extra_1,...,extra_n)
            large_forward: batch_size, 3大anchor框, size1=network_input_size_h / 32, size2=network_input_size_w / 32, no=(x,y,w,h,conf,cls1,...,clsn,extra_1,...,extra_n)
        targets: torch.size = (該batch中的目標數量, [該image屬於該batch的第幾張圖,cls,x,y,w,h,extra_1,...,extra_n])  x,y,w,h,extra_n in normalized format(extra_n 需要的話)


    """

    count_targets(targets, counts_of_each_cls)

    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    
    if loss_terms.find('euler') !=-1:
        leuler, lp_t_red = torch.zeros(1, device=device), torch.zeros(1, device=device)

    lextra_list = [torch.zeros(1, device=device) for _ in range(n_loss_terms-4)]  # n_loss_terms - 4(lbox,lobj,lcls,total_loss)
    
    # if loss_terms.find('dhxdhy') != -1 or loss_terms.find('sincos') != -1:
    #     lextra1, lextra2 = torch.zeros(1, device=device), torch.zeros(1, device=device)
    
    tcls, tbox, indices, anchors, extra_outs, offsets = build_targets(p, targets, model, theta_format, num_extra_outputs)  # targets
 
    # plot anchors on training image and save it in first iteration
    if tb_writer and num_iter == 0:
    # if tb_writer and batch_No == 0 and epoch%1==0:
        save_and_plot_anchors_on_img_to_tensorboard(model, log_dir, num_iter, epoch, cv_imgs, indices, tbox, anchors, offsets, theta_format, extra_outs, tb_writer)
    
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)
    SML1 = nn.SmoothL1Loss(reduction='mean')
    MSE = nn.MSELoss(reduction='mean')

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
    
    # Losses
    iou_method = 'iou'
    if loss_terms.find('giou') != -1: iou_method = 'giou'
    elif loss_terms.find('diou') != -1: iou_method = 'diou'
    elif loss_terms.find('ciou') != -1: iou_method = 'ciou'
    elif loss_terms.find('iou') != -1: iou_method = 'iou'
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if np == 5 else balance
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # print('pi dtype', pi.dtype)
            # print('b dtype', b.dtype)
            # print('a dtype', a.dtype)
            # print('gi dtype', gi.dtype)
            # print('gj dtype', gj.dtype)
            # print('ps dtype', ps.dtype)

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            
            # pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            # pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box

            if loss_terms.find('dhxdhy') != -1:
                pextras = ps[:,-num_extra_outputs:]
            elif loss_terms.find('sincos') != -1:
                pextras = ps[:,-num_extra_outputs:].tanh()

            # pextras = ps[:,-num_extra_outputs:]

            # print('before tanh pextras[:,extra_loss_idx]', pextras)
            
            # get theta
            if theta_format.find('dhxdhy') != -1:
                dx, dy = extra_outs[i][:,0]-tbox[i][:,0], extra_outs[i][:,1]-tbox[i][:,1]
                t_theta = torch.atan2(-dy,dx)
                dx, dy = pextras[:,0]-pbox[:,0], pextras[:,1]-pbox[:,1]
                p_theta = torch.atan2(-dy,dx)
            elif theta_format.find('sincos') != -1:
                t_theta = torch.atan2(extra_outs[i][:,0],extra_outs[i][:,1])
                p_theta = torch.atan2(pextras[:,0],pextras[:,1])

            t_theta = torch.where(t_theta<0, t_theta+2*math.pi, t_theta)
            p_theta = torch.where(p_theta<0, p_theta+2*math.pi, p_theta)

            # 0 <= delta_theta <= pi
            delta_theta = abs(p_theta-t_theta)
            delta_theta = torch.where(delta_theta <= math.pi, delta_theta, 2*math.pi-delta_theta)

            # horizontal box iou loss
            # iou loss 
            if loss_terms.find('h'+iou_method) != -1:
                if loss_terms.find('giou') != -1:
                    giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, GIoU=True)
                elif loss_terms.find('diou') != -1:
                    giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, DIoU=True)
                elif loss_terms.find('ciou') != -1:
                    giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
                elif loss_terms.find('iou') != -1:
                    giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False)
            # rotate iou loss
            elif loss_terms.find('r'+iou_method) != -1:
                # convert (x,y,w,h,theta)->(x,-y,h,w,theta-pi/2 )
                # shapely, differentiable rotate iou都是這樣的格式
                # 角度減90度 y軸顛倒  計算rotate iou前都需要經過轉換
                tbox_cvt = tbox[i].clone()
                tbox_cvt[:,1] = -tbox[i][:,1]
                t_theta = torch.where(t_theta-math.pi/2<0, t_theta-math.pi/2+2*math.pi, t_theta-math.pi/2)
                pbox_cvt = pbox.clone()
                pbox_cvt[:,1] = -pbox[:,1]
                p_theta = torch.where(p_theta-math.pi/2<0, p_theta-math.pi/2+2*math.pi, p_theta-math.pi/2)

                txywhtheta = torch.cat((tbox_cvt,t_theta.view(-1,1)),dim=1).unsqueeze(0)
                pxywhtheta = torch.cat((pbox_cvt,p_theta.view(-1,1)),dim=1).unsqueeze(0)

                # box1 = [0.48657,  0.45166,  8.99808, 26.10970, -2.25768]
                # box2 = [-0.36310,  0.55042,  1.95625,  2.91670, -5.39944]
                # txywhtheta = torch.tensor([[[0.97299,  0.89492,  0.66028,  1.58082, -3.17491]]], device=pbox.device)
                # pxywhtheta = torch.tensor([[[0.82812,  0.64551,  0.72205,  1.38668, -2.85743]]], device=pbox.device)
                # if torch.isnan(pxywhtheta).any():
                #     print('pxywhtheta', pxywhtheta)

                if loss_terms.find('giou') != -1:
                    giou = oriented_iou_loss.cal_giou(pxywhtheta, txywhtheta,epoch=epoch)[0]
                elif loss_terms.find('diou') != -1:
                    giou = oriented_iou_loss.cal_diou(pxywhtheta, txywhtheta,epoch=epoch)[0]
                elif loss_terms.find('iou') != -1:
                    # print('iou loss~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    # print('pxywhtheta.dtype', pxywhtheta.dtype)
                    # print('txywhtheta.dtype', txywhtheta.dtype)
                    giou = oriented_iou_loss.cal_iou(pxywhtheta, txywhtheta, epoch)[0]
                    # print('giou', giou)
                    # exit(0)

            if use_angle_modify_factor:
                lay_lbox = (1.0 - giou*torch.cos(delta_theta/2)).mean()
            else:
                lay_lbox = (1.0 - giou).mean()
            # if (giou>1).any():
            #     # print('gpxywhthetaiou[giou>1] ',pxywhtheta[giou>1])
            #     # print('txywhtheta[giou>1] ',txywhtheta[giou>1])
            #     # print('giou[giou>1] ',giou[giou>1])
            #     with open('check_ioulargerthan1.txt', 'a') as outfile:
            #         for g,p,t,b1,b2,c1,c2,a1,a2,ite,u,io in zip(giou[giou>1], pxywhtheta[giou>1], \
            #                         txywhtheta[giou>1], \
            #                         oriented_iou_loss.check_use_box1[giou>1], \
            #                         oriented_iou_loss.check_use_box2[giou>1], \
            #                         oriented_iou_loss.check_use_corners1[giou>1], \
            #                         oriented_iou_loss.check_use_corners2[giou>1], \
            #                         oriented_iou_loss.check_use_area1[giou>1], \
            #                         oriented_iou_loss.check_use_area2[giou>1], \
            #                         oriented_iou_loss.check_use_inter_area[giou>1], \
            #                         oriented_iou_loss.check_use_u[giou>1], \
            #                         oriented_iou_loss.check_use_iou[giou>1] ):
            #             outfile.write('epoch:{} iou:{} p:{} t:{}\n'.format(epoch,g,p,t))
            #             outfile.write('box1:{} corner1:{} area1:{}\n'.format(b1,c1,a1))
            #             outfile.write('box2:{} corner2:{} area2:{}\n'.format(b2,c2,a2))
            #             outfile.write('inter_area:{} union:{} iou:{}\n'.format(ite,u,io))

                        # oriented_iou_loss.check_use_box1[giou>1]
                        # oriented_iou_loss.check_use_box2[giou>1]
                        # oriented_iou_loss.check_use_corners1[giou>1]
                        # oriented_iou_loss.check_use_corners2[giou>1]
                        # oriented_iou_loss.check_use_area1[giou>1]
                        # oriented_iou_loss.check_use_area2[giou>1]
                        # oriented_iou_loss.check_use_inter_area[giou>1]
                        # oriented_iou_loss.check_use_u[giou>1]
                        # oriented_iou_loss.check_use_iou[giou>1]
                # print('giou', giou)
                # input('iou > 1')
            # print('lay_lbox', lay_lbox)

            lbox += lay_lbox
            # lbox += (1.0 - giou).mean()  # giou loss
            
            # if lbox.isnan():
            #     print('iou nan problem ~~~~~~~~~~~~~~~~~~')
            #     # print('lbox', lbox)
            #     with open('check_iou.txt', 'w') as outfile:
            #         print('txywhtheta', txywhtheta)
            #         print('pxywhtheta', pxywhtheta)
            #         print('giou', giou)
            #         for g,p,t in zip(giou[0], pxywhtheta[0], txywhtheta[0]):
            #             outfile.write('{} {} {}\n'.format(g,p,t))

                #exit(0)

            # Objectness
            if use_angle_modify_factor:
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * (giou*torch.cos(delta_theta/2)).detach().clamp(0).type(tobj.dtype)  # giou ratio    
            else:
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:-num_extra_outputs], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:-num_extra_outputs], t)  # BCE
            
            # Extra loss
            if loss_terms.find('euler') != -1:
                lp_t_red += ((1.- torch.sqrt(pextras[:,0]**2+pextras[:,1]**2))**2).mean()

            # if loss_terms.find('dhxdhy') != -1:
            #     pass
            # elif loss_terms.find('sincos') != -1:
            #     pextras = ps[:,-num_extra_outputs:].tanh()

            # print('after tanh pextras', pextras)
            # print('extra_outs[i]', extra_outs[i])

            # print('testl', testl)  # testl tensor(1.11000, device='cuda:0', grad_fn=<SmoothL1LossBackward>)
            for extra_loss_idx in range(n_loss_terms-4):
                lextra_list[extra_loss_idx] += SML1(pextras[:,extra_loss_idx], extra_outs[i][:,extra_loss_idx])

            # lextra1 += SML1(pextra1, extra_outs[i][:,0])
            # lextra2 += SML1(pextra2, extra_outs[i][:,1])
            # lextra1 += MSE(pextra1, extra_outs[i][:,0])
            # lextra2 += MSE(pextra2, extra_outs[i][:,1])
            # print('lextra1', lextra1)  # lextra1 tensor([1.09165], device='cuda:0', grad_fn=<AddBackward0>)
            
        
            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        # lobj += BCEobj(pi[..., 4], tobj)  # obj loss
        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling
    lbox *= h['giou'] * s
    lobj *= h['obj'] * s * (1.4 if np >= 4 else 1.)
    lcls *= h['cls'] * s

    if loss_terms.find('dhxdhy') != -1:
        # print(' h[dhx]',  h['dhx'])
        # print(' h[dhy]',  h['dhy'])
        # print(' s',  s)
        # lextra1 *= h['dhx'] * s
        # lextra2 *= h['dhy'] * s
        lextra_list[0] *= h['dhx'] * s
        lextra_list[1] *= h['dhy'] * s
    elif loss_terms.find('sincos') != -1:
        lextra_list[0] *= h['sin'] * s
        lextra_list[1] *= h['cos'] * s

    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls

    if loss_terms.find('euler') != -1:
        # loss_im+loss_re+loss_im_re_red  (https://github.com/maudzung/Complex-YOLOv4-Pytorch)
        for extra_loss_idx in range(n_loss_terms-4):
            leuler += lextra_list[extra_loss_idx]
        leuler += lp_t_red
        loss += leuler * h['euler']
    else:
        for extra_loss_idx in range(n_loss_terms-4):
            loss += lextra_list[extra_loss_idx]

    return_loss_terms = torch.cat((lbox, lobj, lcls, torch.stack(lextra_list).squeeze(1), loss))
    return loss * bs, return_loss_terms.detach()

    # loss = lbox + lobj + lcls + lextra1 + lextra2
    # return_loss_terms = torch.cat((lbox, lobj, lcls, lextra1, lextra2, loss))   
    # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model, theta_format='dhxdhy', num_extra_outputs=0):
    """
        Build targets for compute_loss()；

    Input:
        p: [small_forward, medium_forward, large_forward]  eg:small_forward.size=(batch_size, 3種scale anchor框, size1, size2, no)
            small_forward: 3小anchor框, size1=network_input_size_h / 8, size2=network_input_size_w / 8, no=(x,y,w,h,conf,cls1,...,clsn,extra_1,...,extra_n)
            medium_forward: 3中anchor框, size1=network_input_size_h / 16, size2=network_input_size_w / 16, no=(x,y,w,h,conf,cls1,...,clsn,extra_1,...,extra_n)
            large_forward: 3大anchor框, size1=network_input_size_h / 32, size2=network_input_size_w / 32, no=(x,y,w,h,conf,cls1,...,clsn,extra_1,...,extra_n)
        targets: torch.size = (該batch中的目標數量, [該image屬於該batch的第幾個圖,cls,x,y,w,h,extra_1,...,extra_n])  x,y,w,h,extra_n in normalized format(extra_n 需要的話)

    Output:
        tcls:  3個tensor組成的list (tensor_class_list[i]) 對每個步長網路生成對應的class tensor
                    tcls[i].shape=(nt,1)
                    e.g., tcls=[tensor([0,2,7]), tensor([1,2,2]), tensor([3,5,6])]

        tbox:  3個tensor組成的list (box[i]) 對每個步長網路生成對應的gt_box訊息
                xy：當前feature map尺度上的真實gt_xy與負責預測網格座標的偏移量;
                wh：當前feature map尺度上的真實gt_wh
                    tbox[i].shape=(nt,4)
                    e.g., tbox=[tensor([[0.08798, -0.20863,  3.44978, 10.18106],
                                        [0.09541, -0.48024,  0.71561,  1.91806]]),
                                tensor([[0.51959,  0.79366,  1.28252,  3.31385],
                                        [0.63755, -0.15933,  0.36991,  0.99074]]),
                                tensor([[0.95391,  0.65906,  0.63891,  1.60158],
                                        [1.19355,  0.27958,  4.38709, 14.92512]])]

        indices:  索引列表 由3個大tuple組成的list 每個tuple代表對每個步長網路生成的索引數據。其中單個list中的索引數據有：
                    (該image屬於該batch的第幾張圖;該box屬於哪種scale的anchor;網格h索引;網格w索引)
                        indices[i].shape=(4,nt)
                    e.g., indices=[(tensor([0,1]), tensor([0,2]), tensor([200,102]), tensor([180,98])), 
                                    (tensor([0,1,1,2]), tensor([0,0,1,2]), tensor([110,120,80,76]), tensor([56,49,86,43])), 
                                    (tensor([0,0,1,2]), tensor([0,0,1,2]), tensor([59,30,21,33]), tensor([58,30,46,43]))]

        arch:  anchor列表 由3個tensor組成的list 每個tensor代表每個步長網路對gt目標採用的anchor大小(對應feature map尺度上的anchor_wh)
                    anchor[i].shape=(nt,2)
                    e.g., anch=[tensor([[1.37500, 1.37500],[1.37500, 1.37500],[1.37500, 3.25000]]), 
                                tensor([[0.87500, 1.93750],[1.43750, 3.25000],[1.43750, 3.25000]]), 
                                tensor([[0.84375, 2.00000],[0.84375, 2.00000],[1.56250, 6.87500]])]

        extra_outs:  3個tensor組成的list 每個tensor儲存額外channel的數據 如果有長度的值會轉換到當前feature map尺度上
                    extra_outs[i].shape=(nt,?)
    """
    nt = targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch, extra_outs, offsets_list = [], [], [], [], [], []
    gain = torch.ones(targets.shape[1], device=targets.device)  # normalized to gridspace gain
    """
    offset用來決定物體中心的grid上下左右哪些grid也被挑來計算loss，會用gxy去減
    (50.4,50.4)左邊一格為(49.4,50.4)所以offset設為[1, 0]，以此類推
    off依序為左 上 右 下
    """
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

    g = 0.5  # offset  網格中心偏移
    multi_gpu = is_parallel(model)
    # get yolo layer id
    for i, jj in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
        """
        jj 144  small_forward
        jj 159  medium_forward
        jj 174  large_forward
        """

        """
        anchor torch.size=[一層yolo layer的anchor數量, 2(w,h)]  w,h已經除以stride
        """
        anchors = model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec
        anchors = anchors.to(targets.device)
        
        """
        以extra output=2為例, gain=[1,1,w,h,w,h,1,1]
        """
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain for targets

        if theta_format.find('dhxdhy') != -1:
            """
            如果extra output是dhxdhy為例, gain=[1,1,w,h,w,h,w,h]
            """ 
            gain[-num_extra_outputs:] = torch.tensor(p[i].shape)[[3, 2]]

        """
        t normalized back to grid space
        """
        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            """
            GT的wh與anchor的wh做匹配，篩選掉比值大於hyp['anchor_t']的(這應該是yolov5的創新點)targets,從而更好的回歸(與新的邊框回歸方式有關)
            若gt_wh/anchor_wh或anchor_wh/gt_wh太大超出hyp['anchor_t'],則說明當前targets與所選anchor形狀匹配不高,該物體寬高過於極端,不應該強制回歸

            """
            na = anchors.shape[0]  # number of anchors

            """
            at.shape = (na, nt) 生成anchor索引 anchor index; at[0]全等於0 at[1]全等於1 at[2]全等於2 用於表示當前gtbox和當前層哪個anchor匹配
            torch.arrange(na): tensor([0,1,2])  torch.size([3])
            torch.arange(na).view(na, 1): tensor([[0],[1],[2]])  torch.size([3,1])
            torch.arange(na).view(na, 1).repeat(1, nt): if nt=5 tensor([[0,0,0,0,0],[1,1,1,1,1],[2,2,2,2,2]]) torch.size([3,5])
            """
            at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

            """
            GT的wh與anchor的wh做匹配，篩選掉比值大於hyp['anchor_t']的(這應該是yolov5的創新點)targets，
            從而更好的回歸(與新的邊框回歸方式有關)，若gt_wh/anchor_wh 或 anchor_wh/gt_wh 超出hyp['anchor_t']，
            則說明當前target與所選anchor形狀匹配度不高，該物體寬高過於極端，不應該強制回歸，將該處的labels信息刪除，在該層預測中認為是背景

            由於yolov3回歸wh採用的是out=exp(in)，這很危險，因為out=exp(in)可能會無窮大，就會導致失控的梯度，不穩定，NaN損失並最終完全失去訓練；
            當然原yolov3採用的是將targets進行反算來求in與網路輸出的結果，就問題不大，但採用iou loss，就需要將網路輸出算成out來進行loss求解，所以會面臨這個問題；
            所以作者採用新的wh回歸方式：
            (wh.sigmoid()*2)**2*anchor[i]，原來yolov3為exp(wh)*anchor[i]
            將標籤框與anchor的倍數控制在0-4之間；
            hyp.scratch.yaml中的超參數anchor_t=4，所以也是通過此參數來判定anchors與標籤框契合度

            t -> t[None, :, 4:6]   size (nt,8) -> (1,nt,2)
            anchors -> anchors[:, None]   size (na,2) -> (na,1,2)
            r (wh ratio):  torch.size([na,nt,2])

            從r跟1./r中分別取w的max以及h的max
            e.g.,  na=3, nt=1
                r [[[1.13506, 1.89483]]  1./r [[[0.88101, 0.52775]]
                   [[1.38730, 0.99253]]        [[0.72082, 1.00753]]
                   [[1.13506, 0.80166]]]       [[0.88101, 1.24741]]]
            torch.max(r, 1. / r)  torch.size([na,nt,2])
                   [[[1.13506, 1.89483]]
                    [[1.38730, 1.00753]]
                    [[1.13506, 1.24741]]]
            
            torch.max(r, 1. / r).max(2)  max(2):表示取第三個維度的max，會返回最大value跟該value的index
            torch.max(r, 1. / r).max(2)[0] 表示取值的部份  torch.size([na,nt])
            """
            r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare  shape=(3,num_gt_batch)
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))

            """
            t.repeat(na, 1, 1) torch.size([na(原來數據的基礎上重複na次，按row拼街), nt(該batch中gt數量), 8(該image屬於該batch的第幾個圖,cls,x,y,w,h,extra1,extra2)])

            經過第一次長寬比的篩選後
            a shape = (符合條件的target數量)  filter過後gt對應到的anchor id
            t shape = (符合條件的target數量, 8[該image屬於該batch的第幾個圖,cls,x,y,w,h,extra1,extra2])
            """
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            """
            gxy % 1.  <-- 取小數部份
            g=0.5表示以一個grid的中心點劃分為2x2並判斷物體中心點偏在grid的哪一位置(是偏grid內靠上靠左還是靠右靠下)
            因為是看上下左右所以(gxy > 1.)以及(gxy < (gain[[2, 3]] - 1.)是為了確保不要超出grid space size
            把相對於各個網格左上角x<g=0.5, y<g=0.5 (j,k)以及大於x>g=0.5, y>g=0.5 (l,m)的框提出來;
            在選取gij(也就是標籤框分配給的網格的時候)對這四個部份的框都做一個偏移(減去上面的off)，也就是下面的gij=(gxy-offsets).long()操作;
            再將這四個部份的框與原始的gxy拼接在一起，總共就是五個部份;
            也就是說：①將每個網格按照2x2分成四個部份，每個部份的框不僅採用當前網格的anchor進行回歸，也採用該部份相鄰的兩個網格的anchor進行回歸;
            原yolov3就僅僅採用當前網格的anchor進行回歸;
            估計是用來緩解網格效應，但由於v5沒發論文，所以也只是推測，yolov4也有相關解決網格效應的措施，是通過對sigmoid輸出乘以一個大於1的係數;
            這也與yolov5新的邊框回歸公式相關;
            由於①，所以中心點回歸也從yolov3的0~1的範圍變成-0.5~1.5的範圍;
            所以中心點回歸的公式變為;
            xy.sigmoid()*2.-0.5+cx
            """
            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)

            #  j,k 判斷的是v，多增加1跟2的框     l,m 判斷的是v，多增加3跟4的框
            #  _______                        _______
            #  |  |11|                        | v|33|
            #  |__|__|                        |vv|  |
            #  |22|vv|                        |‾‾|‾‾|
            #  |  |v |                        |44|  |
            #  ‾‾‾‾‾‾‾                        ‾‾‾‾‾‾‾
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 提取篩選後的gt中心座標相對於各個網格的中心點xy偏移<0.5 同時>1
            l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T  # 提取篩選後的gt中心座標相對於各個網格的中心點xy偏移>0.5 同時<grid size-1
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g
            
        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        if theta_format.find('dhxdhy') != -1:
            gextra_outs = t[:,-num_extra_outputs:] - gij  # 物體頭點與grid左上角點的偏移量
        elif theta_format.find('sincos') != -1:
            gextra_outs = t[:,-num_extra_outputs:]
        else:
            print('No supported train mode')
            exit(0)

        # Append
        #indices.append((b, a, gj, gi))  # image, anchor, grid indices
        # 直接將gi gj超出範圍的值鎖在範圍內
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        extra_outs.append(gextra_outs)  # extra channel
        offsets_list.append(offsets)

    return tcls, tbox, indices, anch, extra_outs, offsets_list


def rotate_non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, \
                            classes=None, agnostic=False, theta_format='dhxdhy', \
                            num_extra_outputs=0, whichdataset='vehicle8cls'):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Input:
        prediction shape: torch.size([num_img_in_batch, num_anchor*(64*64+32*32+16*16=16*16*(16+4+1)), 7+cls(x,y,w,h,conf,cls1,...,clsn,extra1,extra2)])

    Output:
        len: N images
                [image1_nms_result, iamge2_nms_result, ..., imageN_nms_result]
        nms_result shape: m rboxes, [m,8] in torch size
                [[xc,yc,w,h,conf,cls_idx,xh,yh]
                [xc,yc,w,h,conf,cls_idx,xh,yh]
                .
                .
                ]
    """

    # nms_iou_thres_dota10 = {'plane': 0.3, 'baseball-diamond': 0.3, 'bridge': 0.0001, 'ground-track-field': 0.3, 
    #             'small-vehicle': 0.2, 'large-vehicle': 0.1, 'ship': 0.05, 'tennis-court': 0.3, 
    #             'basketball-court': 0.3, 'storage-tank': 0.2, 'soccer-ball-field': 0.3, 'roundabout': 0.1, 
    #             'harbor': 0.0001, 'swimming-pool': 0.1, 'helicopter': 0.2}

    nms_iou_thres_dota10 = [('plane',0.3),('baseball-diamond',0.3),('bridge',0.0001),
                            ('ground-track-field',0.3),('small-vehicle',0.2),
                            ('large-vehicle',0.1),('ship',0.05),('tennis-court',0.3),
                            ('basketball-court',0.3),('storage-tank',0.2),
                            ('soccer-ball-field',0.3),('roundabout',0.1),('harbor',0.0001),
                            ('swimming-pool',0.1),('helicopter',0.2)]

    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5 - num_extra_outputs # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 3000  # maximum number of detections per image  default 300
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = False  # multiple labels per box (adds 0.5ms/img)
    # 如果一種物體有多個標籤，由於這邊的作法是直接多一個框，如果有三種標籤同一個物體就多三個框
    # 框的數量級增加太快這樣在算torchvision.ops.boxes.nms很容易gpu out of memory，需要修改
    # multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:-num_extra_outputs] *= x[:, 4:5]  # conf = cls_conf * obj_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # box = xywh2xyxy(x[:, :4])

        # Box (center x, center y, width, height) to (x, y, w, h, θ)
        if theta_format == 'dhxdhy':
            dx, dy = x[:, -num_extra_outputs]-x[:,0], x[:, -num_extra_outputs+1]-x[:,1]
            theta = torch.atan2(-dy, dx)
        elif theta_format == 'sincos':
            theta = torch.atan2(x[:, -num_extra_outputs], x[:, -num_extra_outputs+1])

        theta = torch.where(theta<0, theta+2*math.pi, theta)
        box = torch.cat((x[:,:4], theta.view(-1,1)), dim=1)

        # Detections matrix nx6 (x,y,w,h,θ,conf,cls)
        if multi_label:
            """
            nonzero: 取出每個軸的索引，末認識非0元素的索引(取出括號公式中的為True的元素對應的索引) 將索引放入i,j中
            i: num_boxes該維度中的索引號，表示該索引的box其中有class的conf滿足要求 length=x中滿足條件的所有座標數量
            j: num_classes該維度中的索引號，表示某個box中是第j+1類物體的conf滿足要求 length=x中滿足條件的所有座標數量

            # 第一個box如果有兩類大於conf，則第一個box轉成output格式時會變成兩個框
            # 假設第一個box第一類跟第五類都大於thres
            # e.g., output_x: [[100, 20, 20, 40, 2.87, conf1, cls1]
            #                  [100, 20, 20, 40, 0.35, conf5, cls5]]
            """
            i, j = (x[:, 5:-num_extra_outputs] > conf_thres).nonzero(as_tuple=False).T
            # 按行並接 list x:(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ∈[0,2pi]
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:-num_extra_outputs].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes:
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]
        
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # 0<=x1 0<=y1
        # x = x[(x[:,[0,1]]>0).all(1)]

        # 如果box太多可能cuda out of memory

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        if n > 100000:
            # print('Number of box: {}, avoid gpu out of memory, skip nms'.format(n))
            continue
        
        # else:
            # print('Number of box: {}, to nms'.format(n))

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        # 此處nms作法是將不同類別的框分開計算nms，會將不同類別的框向右移動 第n類*max_wh的值 來將不同類別的框分開
        # 如果使用agnostic則是視所有框為同一類做nms
        # x:(num_confthres_boxes, [x,y,w,h,θ,conf,classid]) θ∈[0,2pi]
        # c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        boxes = x[:,:5].clone()
        boxes[:,:2] += c

        # convert (xywhtheta)->(x-yhwtheta) 
        # shapely, differentiable rotate iou都是這樣的格式
        # 計算rotate iou前都需要經過轉換
        # 角度減90度 y軸顛倒  計算rotate iou前都需要經過轉換
        boxes4nms = boxes.clone()
        boxes4nms[:,1] = -boxes[:,1]
        boxes4nms[:,4] = torch.where(boxes4nms[:,4]-math.pi/2<0, boxes4nms[:,4]-math.pi/2+2*math.pi, boxes4nms[:,4]-math.pi/2)
 
        scores = x[:, 5]

        # Trying to create tensor with negative dimension 如果box太多會有這個error
        # i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)

        # if whichdataset == 'dota1.0': # is dota1.0 dataset
        #     temp_output_list = []
        #     for cls_tensor in x[:,6].unique():
        #         pcls_mask = (x[:,6] == cls_tensor).nonzero().view(-1)
        #         i = rotate_gpu_nms(boxes4nms[pcls_mask].cpu().numpy(), nms_iou_thres_dota10[int(cls_tensor)][1], torch.cuda.current_device())
        #         temp_output_list.append(x[pcls_mask][i])
                
        #     temp_out = torch.cat(temp_output_list)

        #     output[xi] = temp_out
        # else:        
        i = rotate_gpu_nms(torch.cat((boxes4nms,scores.view(-1,1)),dim=1).cpu().numpy(), iou_thres, torch.cuda.current_device())
        # i = nms_rotate_cpu(boxes.cpu().numpy(), scores.cpu().numpy(), iou_thres)
        i = torch.from_numpy(i)
        

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
    
    return output


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False, num_extra_outputs=0):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Input:
        prediction shape: torch.size([num_img_in_batch, num_anchor*(64*64+32*32+16*16=16*16*(16+4+1)), 7+cls(x,y,w,h,conf,hx,hy,n_classes)])

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5 - num_extra_outputs # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = False  # multiple labels per box (adds 0.5ms/img)
    # 如果一種物體有多個標籤，由於這邊的作法是直接多一個框，如果有三種標籤同一個物體就多三個框
    # 框的數量級增加太快這樣在算torchvision.ops.boxes.nms很容易gpu out of memory，需要修改
    # multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:-num_extra_outputs] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:-num_extra_outputs] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:-num_extra_outputs].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # 0<=x1 0<=y1
        # x = x[(x[:,[0,1]]>0).all(1)]

        # 如果box太多可能cuda out of memory

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        # if not n or n > 100000:
        #     print('Number of box: {}, skip nms'.format(n))
        #     continue
        # else:
        #     print('Number of box: {}, to nms'.format(n))

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        # Trying to create tensor with negative dimension 如果box太多會有這個error
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def v5obb_rotate_non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False, without_iouthres=False, num_extra_outputs=0):
    """
    Performs Rotate-Non-Maximum Suppression (RNMS) on inference results；

    Input:
        prediction shape: torch.size([num_img_in_batch, num_anchor*(64*64+32*32+16*16=16*16*(16+4+1)), 7+cls(x,y,w,h,conf,hx,hy,n_classes)])

    @param prediction: size=(batch_size, num, [xywh,score,num_classes,extra1,...])
    @param conf_thres: 置信度阈值
    @param iou_thres:  IoU阈值
    @param merge: None
    @param classes: None
    @param agnostic: 进行nms是否将所有类别框一视同仁，默认False
    @param without_iouthres : 本次nms不做iou_thres的标志位  默认为False
    @return:
            output：经nms后的旋转框(batch_size, num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]
    """
    # prediction :(batch_size, num_boxes, [xywh,score,num_classes,num_angles])
    nc = prediction[0].shape[1] - 5 - num_extra_outputs  # number of classes = no - 5 -180
    class_index = nc + 5
    # xc : (batch_size, num_boxes) 对应位置为1说明该box超过置信度
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections 要求冗余检测
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    # output: (batch_size, ?)
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x ： (num_boxes, [xywh, score, num_classes, num_angles])
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # 取出数组中索引为True的的值即将置信度符合条件的boxes存入x中   x -> (num_confthres_boxes, no)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        angle = x[:, class_index:]  # angle.size=(num_confthres_boxes, [num_angles])
        # torch.max(angle,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        angle_value, angle_index = torch.max(angle, 1, keepdim=True)  # size都为 (num_confthres_boxes, 1)
        # box.size = (num_confthres_boxes, [xywhθ])  θ∈[0,179]
        box = torch.cat((x[:, :4], angle_index), 1)
        if multi_label:
            # nonzero ： 取出每个轴的索引,默认是非0元素的索引（取出括号公式中的为True的元素对应的索引） 将索引号放入i和j中
            # i：num_boxes该维度中的索引号，表示该索引的box其中有class的conf满足要求  length=x中满足条件的所有坐标数量
            # j：num_classes该维度中的索引号，表示某个box中是第j+1类物体的conf满足要求  length=x中满足条件的所有坐标数量
            i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T
            # 按列拼接  list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ∈[0,179]
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if without_iouthres:  # 不做nms_iou
            output[xi] = x
            continue

        # Filter by class 按类别筛选
        if classes:
            # list x：(num_confthres_boxes, [xywhθ,conf,classid])
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)] # any（1）函数表示每行满足条件的返回布尔值

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]
        # Batched NMS
        # x：(num_confthres_boxes, [xywhθ,conf,classid]) θ∈[0,179]
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        # boxes:(num_confthres_boxes, [xy])  scores:(num_confthres_boxes, 1)
        # agnostic用于 不同类别的框仅跟自己类别的目标进行nms   (offset by class) 类别id越大,offset越大
        boxes_xy, box_whthetas, scores = x[:, :2] + c, x[:, 2:5], x[:, 5]
        rects = []
        for i, box_xy in enumerate(boxes_xy):
            rect = longsideformat2poly(box_xy[0], box_xy[1], box_whthetas[i][0], box_whthetas[i][1], box_whthetas[i][2])
            rects.append(rect)
        i = np.array(py_cpu_nms_poly(np.array(rects), np.array(scores.cpu()), iou_thres))
        #i = nms(boxes, scores)  # i为数组，里面存放着boxes中经nms后的索引

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]  # 根据nms索引提取x中的框  x.size=(num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]

        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def strip_optimizer(f='weights/best.pt', s=''):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    #x['model'].half()  # to FP16
    #for p in x['model'].parameters():
    #    p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/train2017/'):  # from utils.utils import *; coco_only_people()
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def crop_images_random(path='../images/', scale=0.50):  # from utils.utils import *; crop_images_random()
    # crops images into random squares up to scale fraction
    # WARNING: overwrites images!
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        img = cv2.imread(file)  # BGR
        if img is not None:
            h, w = img.shape[:2]

            # create random mask
            a = 30  # minimum size (pixels)
            mask_h = random.randint(a, int(max(a, h * scale)))  # mask height
            mask_w = mask_h  # mask width

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.utils import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def output_to_target(output, width, height):
    """
    Input:
        output: [x,y,w,h,θ,conf,classid]
    Return:
        [batch_id, class_id, x, y, w, h, θ, conf]
    """
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    # num_extra_outputs = output.shape[1]
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                # box = pred[:4]
                # w = (box[2] - box[0]) / width
                # h = (box[3] - box[1]) / height
                # x = box[0] / width + w / 2
                # y = box[1] / height + h / 2
                # conf = pred[4]
                # cls = int(pred[5])

                x,y,w,h,theta = pred[:5]                
                conf = pred[5]
                cls = int(pred[6])               

                targets.append([i, cls, x, y, w, h, theta, conf])

    return np.array(targets)


def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):len(dir)+Path(x).name.find('_')-len(Path(dir).name) if '_' in Path(x).name else None]) for x in d]) + 1  # increment
        # n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


# Plotting functions ---------------------------------------------------------------------------------------------------
def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def save_and_plot_valid_result_to_tensorboard(f, preds, conf_thres, theta_format, num_extra_outputs, img, save_dir, epoch, tb_writer):
    """
    preds:
        from inf_out: (num_img_in_batch, anchors*grid_h*frid_w, 15[x,y,w,h,conf,cls1,...,clsn,extra1,...])
        from nms_out: (num_img_in_batch, 7[x,y,w,h,θ,conf,classid])
    """

    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    x = preds[0]  # 只存batch中第一張
    
    if x is None: return None

    nms_out = x.shape[1] == 7  # [x,y,w,h,θ,conf,classid]
    conf = x[:,5] if nms_out else x[:,4]  # check for confidence presence (gt vs pred)

    x = x[conf>conf_thres]
    # x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

    if x.shape[0] > 0:
        if theta_format.find('dhxdhy') != -1:
            dhx,dhy = x[:, -num_extra_outputs]-x[:,0], x[:, -num_extra_outputs+1]-x[:,1]
            theta = torch.atan2(-dhy,dhx)
        elif theta_format.find('sincos') != -1:
            theta = torch.atan2(x[:, -num_extra_outputs], x[:, -num_extra_outputs+1])

        theta = torch.where(theta<0, theta+2*math.pi, theta)
        rbox_xywhtheta = (torch.cat((x[:,:4],theta.view(-1,1)),dim=1).cpu().numpy())
        # print('in general.py save_and_plot_valid_result_to_tensorboard rbox dtype', rbox_xywhtheta.dtype)
        if nms_out: rbox_4xy = xywhtheta24xy_new(x[:,:5])
        else: rbox_4xy = xywhtheta24xy_new(rbox_xywhtheta)

        for idx, pts in enumerate(rbox_4xy):
            if nms_out: draw_one_polygon(img, pts, int(x[idx,6]), True, 1)
            else: draw_one_polygon(img, pts, 4, True, 1)  # before nms, box no class defined, assign color by color_board

    cv2.imwrite(str(save_dir)+os.sep+f, img)    
    # result = plot_images(images=img, targets=targets, paths=paths, fname=f, max_size=1024, theta_format=theta_format, num_extra_outputs=num_extra_outputs, plotontb=True)
    tb_writer.add_image(f, img, dataformats='HWC', global_step=epoch)


def save_and_plot_anchors_on_img_to_tensorboard(model, log_dir, num_iter, epoch, cv_imgs, indices, tbox, anchors, offsets, theta_format, extra_outs, tb_writer):
    multi_gpu = is_parallel(model)
    yolo_layer_id_list = model.module.yolo_layers if multi_gpu else model.yolo_layers
    anchor_stride_list = [model.module.module_list[yolo_layer_id].stride if multi_gpu else model.module_list[yolo_layer_id].stride for yolo_layer_id in yolo_layer_id_list]
    all_anchors_filename = str(log_dir / 'train_batch{}_all_anchors.jpg'.format(num_iter))  # filename

    for img_id, img in enumerate(cv_imgs):
        img_show_all_anchors = copy.deepcopy(img)

        for i, anchor_stride in enumerate(anchor_stride_list):
            img_show_one_anchor = copy.deepcopy(img)
            b, a, gj, gi = indices[i]
            yolo_layer_anchor_filename = str(log_dir / 'train_batch{}_yolo_layer{}_anchors.jpg'.format(num_iter,i))  # filename
            for tbox_id, (box,anchor_wh,offset) in enumerate(zip(tbox[i], anchors[i], offsets[i])):
                if b[tbox_id] == img_id:
                    x, y, w, h = box
                    w = anchor_wh[0]
                    h = anchor_wh[1]
                    x = (x+gi[tbox_id]-offset[0])*anchor_stride
                    y = (y+gj[tbox_id]-offset[1])*anchor_stride
                    w *= anchor_stride
                    h *= anchor_stride

                    if theta_format.find('dhxdhy') != -1:
                        hx = (extra_outs[i][tbox_id][0]+gi[tbox_id]-offset[0])*anchor_stride
                        hy = (extra_outs[i][tbox_id][1]+gj[tbox_id]-offset[1])*anchor_stride
                        theta = torch.atan2(-(hy-y),hx-x)
                    elif theta_format.find('sincos') != -1:
                        theta = torch.atan2(extra_outs[i][tbox_id][0], extra_outs[i][tbox_id][1])
                    else:
                        print('Unsupport theta format!!')
                        exit(0)

                    theta = torch.where(theta<0, theta+2*math.pi, theta)

                    rbox = torch.stack((x,y,w,h,theta))
                    # print('in general.py save_and_plot_anchors_on_img_to_tensorboard rbox dtype', rbox.dtype)
                    rbox_pts = xywhtheta24xy_new(rbox[None,:].cpu().numpy()).squeeze()

                    if i == 0: color = (255,0,0)
                    elif i == 1: color = (0,255,0)
                    elif i == 2: color = (0,0,255)

                    cv2.line(img_show_one_anchor, (int(rbox_pts[0]), int(rbox_pts[1])), (int(rbox_pts[2]), int(rbox_pts[3])), color, 1, cv2.LINE_AA)
                    cv2.line(img_show_one_anchor, (int(rbox_pts[2]), int(rbox_pts[3])), (int(rbox_pts[4]), int(rbox_pts[5])), color, 1, cv2.LINE_AA)
                    cv2.line(img_show_one_anchor, (int(rbox_pts[4]), int(rbox_pts[5])), (int(rbox_pts[6]), int(rbox_pts[7])), color, 1, cv2.LINE_AA)
                    cv2.line(img_show_one_anchor, (int(rbox_pts[6]), int(rbox_pts[7])), (int(rbox_pts[0]), int(rbox_pts[1])), color, 1, cv2.LINE_AA)

                    cv2.line(img_show_all_anchors, (int(rbox_pts[0]), int(rbox_pts[1])), (int(rbox_pts[2]), int(rbox_pts[3])), color, 1, cv2.LINE_AA)
                    cv2.line(img_show_all_anchors, (int(rbox_pts[2]), int(rbox_pts[3])), (int(rbox_pts[4]), int(rbox_pts[5])), color, 1, cv2.LINE_AA)
                    cv2.line(img_show_all_anchors, (int(rbox_pts[4]), int(rbox_pts[5])), (int(rbox_pts[6]), int(rbox_pts[7])), color, 1, cv2.LINE_AA)
                    cv2.line(img_show_all_anchors, (int(rbox_pts[6]), int(rbox_pts[7])), (int(rbox_pts[0]), int(rbox_pts[1])), color, 1, cv2.LINE_AA)

            cv2.imwrite(yolo_layer_anchor_filename, img_show_one_anchor)
            
    if num_iter == 0: cv2.imwrite(all_anchors_filename, img_show_all_anchors)
    tb_writer.add_image('train_batch_all_anchors.jpg', img_show_all_anchors, dataformats='HWC', global_step=epoch)


def draw_one_polygon(img, pts, which_cls=0, rgb_color=False, line_thickness=1):
    """
    draw one polygon on image, head line is white
    image will be modified becuase of reference calling
    you can receive return image if necessary

    Input:
        img: opencv format image
        which_cls: int value
        pts: (x1,y1,x2,y2,x3,y3,x4,y4) should in type numpy ndarray or torch tensor for check nan
        rgb_color: if polygon is displayed on tensorboard, color channel will convert from bgr to rgb

    Output:
        img: opencv format image
    """
    
    # cannot convert float Nan to integer
    # if None in pts:
    #     return None

    for p in pts:
        if isinstance(p, np.ndarray):
            if np.isnan(p).any():
                print('pts ', pts)
                exit(0)
            elif np.isnan(p):
                print('pts ', pts)
                exit(0)
        elif isinstance(p, torch.Tensor):
            if torch.isnan(p).any(): 
                print('pts ', pts)
                exit(0)
            elif torch.isnan(p): 
                print('pts ', pts)
                exit(0)

    # cannot convert float Nan to integer
    # if np.isnan(pts).any():
    #     return

    int_pts = [int(p) for p in pts]
    p1, p2, p3, p4 = (int_pts[0], int_pts[1]), (int_pts[2], int_pts[3]), \
                    (int_pts[4], int_pts[5]), (int_pts[6], int_pts[7])

    # 可以增加 e.g.,[0,0.3,0.5,0.7]  [1,0.7,0.5,0.3]
    bgr_weight_list = [0,0.5,0.7]
    bgr_weight_list2 = [1,0.7,0.5]

    # 不可變
    bgr_bit_list = [[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0],[1,0,1]]

    total_num_colors = len(bgr_bit_list)*len(bgr_weight_list)

    # 可以增加 e.g.,[255,245,235,225,215]
    start_bgr_list = [250, 200, 150]
    start_bgr_idx = (which_cls // total_num_colors) % len(start_bgr_list)

    which_cls = which_cls - total_num_colors * (which_cls//total_num_colors)
    which_color = which_cls % len(bgr_bit_list)

    if sum(bgr_bit_list[which_color]) == 1:
        bgr_bit_is_one_idx = bgr_bit_list[which_color].index(1)
        pre_bgr_bit_idx = bgr_bit_is_one_idx-1
        bgr_color_bit = bgr_bit_list[which_color]
        which_weight = (which_cls//len(bgr_bit_list)) % len(bgr_weight_list)

        if which_cls % total_num_colors >= len(bgr_bit_list):
            bgr_color_bit[pre_bgr_bit_idx] = bgr_weight_list[int(which_weight)]
    elif sum(bgr_bit_list[which_color]) == 2:
        bgr_color_bit = bgr_bit_list[which_color]
        if bgr_color_bit == [0,1,1]: pos_bgr_bit_idx = 2
        elif bgr_color_bit == [1,1,0]: pos_bgr_bit_idx = 1
        elif bgr_color_bit == [1,0,1]: pos_bgr_bit_idx = 0

        which_weight = (which_cls//len(bgr_bit_list)) % len(bgr_weight_list2)

        if which_cls % total_num_colors >= len(bgr_bit_list):
            bgr_color_bit[pos_bgr_bit_idx] *= bgr_weight_list2[int(which_weight)]

    body_line_color = [int(bgr_color*start_bgr_list[start_bgr_idx]) for bgr_color in bgr_color_bit]
    head_line_color = (255,255,255)

    body_line_color = body_line_color[::-1] if rgb_color else body_line_color

    cv2.line(img, p1, p2, head_line_color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, p2, p3, body_line_color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, p3, p4, body_line_color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, p4, p1, body_line_color, line_thickness, lineType=cv2.LINE_AA)

    return img


def plot_one_polygon(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    draw_one_polygon(img, )
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='YOLO')
    plt.plot(x, yb ** 2, '.-', label='YOLO ^2')
    plt.plot(x, yb ** 1.6, '.-', label='YOLO ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16, theta_format='dhxdhy', num_extra_outputs=0, plotontb=False, isGT=True):
    """
    plotontb: if true, images are shown on tensorboard
    """

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
 
    # if os.path.isfile(fname):  # do not overwrite
    #     return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # train.py的時候正常但是test.py的時候array內存有tensor，所以要把裡面所有tensor轉為array
    t = []
    for target in targets:
        t.append([t_item.cpu().numpy() if isinstance(t_item, torch.Tensor) else t_item for t_item in target])
    targets = np.array(t)
    
    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    ori_h, ori_w = h, w
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    # 可能因為matplotlib版本不同，遇到TypeError: int() can't convert non-string with explicit base
    # 如果遇到這個錯誤可以改成下面試試
    # https://github.com/ultralytics/yolov5/issues/2066
    # color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]
    color_lut = [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            classes = image_targets[:, 1].astype('int')
            # gt: (img_id,cls,x,y,w,h,extra1,...)
            # pred: (img_id,cls,x,y,w,h,theta,conf)
            # gt = image_targets.shape[1] == (6+num_extra_outputs)  # ground truth if no conf column
            conf = None if isGT else image_targets[:, 7]  # check for confidence presence (gt vs pred)

            if isGT:
                if theta_format.find('dhxdhy') != -1:
                    dxy = image_targets[:, -num_extra_outputs:]-image_targets[:, 2:4]
                    theta = np.arctan2(-dxy[:,1],dxy[:,0])
                elif theta_format.find('sincos') != -1:
                    theta = np.arctan2(image_targets[:,-num_extra_outputs], image_targets[:,-num_extra_outputs+1])
                else:
                    print('No supported train mode')
                    exit(0)

                theta = np.where(theta<0, theta+2*math.pi, theta)
                rbox = np.concatenate((image_targets[:, 2:6],theta.reshape(-1,1)),axis=1)
                rbox[:,[0,2]] *= w
                rbox[:,[0]] += block_x
                rbox[:,[1,3]] *= h
                rbox[:,[1]] += block_y
            else: 
                rbox = image_targets[:,2:7]
                rbox[:,[0,2]] *= w/ori_w
                rbox[:,[0]] += block_x
                rbox[:,[1,3]] *= h/ori_h
                rbox[:,[1]] += block_y

            # if isGT: print('gt rbox', rbox)
            # else: print('pred rbox', rbox)
            # print('images.shape', images.shape)
            # print('h,w', h,w)

            rbox4pts = xywhtheta24xy_new(rbox)

            for idx, (cls, pts) in enumerate(zip(classes, rbox4pts)):
                if isGT or conf[idx] > 0.3:  # 0.3 conf thresh
                    draw_one_polygon(mosaic, pts, cls, rgb_color=plotontb)

            # boxes = xywh2xyxy(image_targets[:, 2:6]).T
            # boxes[[0, 2]] *= w
            # boxes[[0, 2]] += block_x
            # boxes[[1, 3]] *= h
            # boxes[[1, 3]] += block_y
            # for j, box in enumerate(boxes.T):
            #     cls = int(classes[j])
            #     color = color_lut[cls % len(color_lut)]
            #     cls = names[cls] if names else cls
            #     if gt or conf[j] > 0.3:  # 0.3 conf thresh
            #         label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
            #         plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        # mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        mosaic = cv2.resize(mosaic, (int(ns * w), int(ns * h)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    # optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)


def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_study_txt(f='study.txt', x=None):  # from utils.utils import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in ['coco_study/study_coco_yolov4%s.txt' % x for x in ['s', 'm', 'l', 'x']]:
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        for i in range(7):
            ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
            ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, :j], y[3, :j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=Path(f).stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [33.8, 39.6, 43.0, 47.5, 49.4, 50.7],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid()
    ax2.set_xlim(0, 30)
    ax2.set_ylim(28, 50)
    ax2.set_yticks(np.arange(30, 55, 5))
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig('study_mAP_latency.png', dpi=300)
    plt.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_labels(labels, save_dir=''):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    _,_,patches = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)

    with open(str(save_dir)+os.sep+'label_class_count.txt', 'w') as outfile:
        for i, pp in enumerate(patches):
            outfile.write('{} {}'.format(i, pp._y1))
            if i < len(patches)-1: outfile.write('\n')

    # show each of class number on image
    # for pp in patches:
    #     print('pp',pp)
    #     x = (pp._x0 + pp._x1)/2
    #     y = pp._y1+0.05
    #     ax[0].text(x,y,pp._y1)
    
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig(Path(save_dir) / 'labels.png', dpi=200)
    plt.close()


def plot_evolution(yaml_file='runs/evolve/hyp_evolved.yaml'):  # from utils.utils import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 10), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(5, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
                # y_smooth = butter_lowpass_filtfilt(y)
                # ax[i].plot(x, np.gradient(y_smooth), marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=(), labels=(),
                 save_dir=''):  # from utils.utils import *; plot_results()
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov3
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        files = glob.glob(str(Path(save_dir) / 'results*.txt')) + glob.glob('../../Downloads/results*.txt')
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else Path(f).stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except:
            print('Warning: Plotting error for %s, skipping file' % f)

    fig.tight_layout()
    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)


def get_number_of_extra_outputs(theta_format):
    """
        original yolov4 output
        x,y,w,h,confidence,cls1...clsN
        
        add output channel
        x,y,w,h,confidence,cls1...clsN,extra_output1,...,extra_outputN
    """
    if theta_format.find('sincos') != -1 or theta_format.find('dhxdhy') != -1 or theta_format.find('eular') != -1:
        return 2
    elif theta_format.find('csl') != -1:
        return 0
    else:
        print('No supported train mode!')
        exit(0)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def get_number_of_loss_terms(loss_terms):
    """
    default: 3 (obj loss, class loss, total loss)
    train_mode including:
        'xy' -> loss term +2
        'wh' -> loss term +2
        'dxdy' -> loss term +2
        'iou' -> loss term +1
        'sincos' -> loss term +2
    """
    n_loss_terms = 3  # default: lobj, lcls, total loss
    if loss_terms.find('xy') != -1: n_loss_terms+=2
    if loss_terms.find('wh') != -1: n_loss_terms+=2
    if loss_terms.find('dhxdhy') != -1: n_loss_terms+=2
    if loss_terms.find('iou') != -1: n_loss_terms+=1
    if loss_terms.find('sincos') != -1: n_loss_terms+=2
    if loss_terms.find('distc') != -1: n_loss_terms+=1
    if loss_terms.find('euler') != -1: n_loss_terms+=1

    return n_loss_terms


def print_pbar_title(loss_terms):

    print(('\n' + '%10s' * 2) % ('Epoch', 'gpu_mem'), end='')
    if loss_terms.find('giou') != -1: print(('%10s' * 1) % ('GIoU'), end='')
    elif loss_terms.find('diou') != -1: print(('%10s' * 1) % ('DIoU'), end='')
    elif loss_terms.find('ciou') != -1: print(('%10s' * 1) % ('CIoU'), end='')
    elif loss_terms.find('iou') != -1: print(('%10s' * 1) % ('IoU'), end='')

    print(('%10s' * 2) % ('obj', 'cls'), end='')

    if loss_terms.find('xy') != -1: print(('%10s' * 2) % ('x', 'y'), end='')
    if loss_terms.find('wh') != -1: print(('%10s' * 2) % ('w', 'h'), end='')
    if loss_terms.find('dhxdhy') != -1: print(('%10s' * 2) % ('dhx', 'dhy'), end='')
    if loss_terms.find('sincos') != -1: print(('%10s' * 2) % ('sin', 'cos'), end='')
    if loss_terms.find('euler') != -1: print(('%10s' * 1) % ('euler'), end='')

    print(('%10s' * 3) % ('total', 'targets', 'img_size'))



def r_scale_coords_new(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    # clip_coords(coords, img0_shape)

    # print('img0_shape', img0_shape)
    # input('r scale coords new pause')
    # clip_rboxes(coords, (0, 0), (img0_shape[1]-1, img0_shape[0]-1))
    
    return coords



