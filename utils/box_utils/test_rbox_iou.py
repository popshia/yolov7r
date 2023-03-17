import time
#import tensorflow as tf
import math
#from libs.box_utils.coordinate_convert import *
from rbbox_overlaps import rbbx_overlaps
#from box_utils.rbbox_overlaps import rbbx_overlaps
#from libs.box_utils.iou_cpu import get_iou_matrix

import numpy as np
import cv2



# n box1, m box2, return nxm
def iou_rotate_calculate1(boxes1, boxes2, use_gpu=True, gpu_id=0):

    # start = time.time()
    if use_gpu:
        ious = rbbx_overlaps(boxes1, boxes2, gpu_id)
    else:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        ious = []
        for i, box1 in enumerate(boxes1):
            temp_ious = []
            """Rotated bounding box code"""
            # change angle from radian to degree
            r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4]*180/math.pi)
            for j, box2 in enumerate(boxes2):
                r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4]*180/math.pi)

                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                    temp_ious.append(inter)
                else:
                    temp_ious.append(0.0)
            ious.append(temp_ious)

    # print('{}s'.format(time.time() - start))

    return np.array(ious, dtype=np.float32)

# n box1, n box2, return nx1
def iou_rotate_calculate2(boxes1, boxes2):
    ious = []
    if boxes1.shape[0] != 0:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]

        for i in range(boxes1.shape[0]):
            temp_ious = []
            r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][2], boxes1[i][3]), boxes1[i][4]*180/math.pi)
            r2 = ((boxes2[i][0], boxes2[i][1]), (boxes2[i][2], boxes2[i][3]), boxes2[i][4]*180/math.pi)

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[i] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
            ious.append(temp_ious)

    return np.array(ious, dtype=np.float32)




import shapely.geometry
import shapely.affinity
import math

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle, use_radians=True):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.use_radians = use_radians
        # self.angle = angle if use_radians else angle*180/math.pi
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle, use_radians=self.use_radians)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())


if __name__ == '__main__':
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '13'
    # boxes1 = np.array([[50, 50, 10, 70, -45],
    #                    [150, 150, 10, 50, -50]], np.float32)

    # boxes2 = np.array([[150, 150, 10, 70, -50],
    #                    [150, 150, 10, 70, -50]], np.float32)

    boxes1 = np.array([[50, 50, 100, 50, 0]], np.float32)
    # boxes2 = np.array([[50, 50, 100, 50, 3.1415]], np.float32)
    boxes2 = np.array([[50, 50, 100, 50, 1.57075]], np.float32)

    # total_1 = 0.0
    # total_2 = 0.0
    # n=10000
    # for i in range(n):
    #     start = time.time()
    #     iou = iou_rotate_calculate1(boxes1, boxes2, True, 1)
    #     # print('cal1: ', iou)
    #     total_1 += (time.time() - start)
    #     # print('gpu cost: {}s'.format(time.time() - start))

    #     start = time.time()
    #     iou = iou_rotate_calculate1(boxes1, boxes2)
    #     # print('cal1: ', iou)
    #     total_2 += (time.time() - start)
    #     # print('cpu cost: {}s'.format(time.time() - start))

    # print('gpu cost: {}s'.format(total_1/n))
    # print('cpu cost: {}s'.format(total_2/n))

    iou = iou_rotate_calculate1(boxes1, boxes2)
    print('iou_rotate_calculate1 iou', iou)
    # iou_rotate_calculate1(boxes1, boxes2)


    rr1 = (50, 50, 100, 50, 0)
    rr2 = (50, 50, 100, 50, 90)

    r1 = RotatedRect(*rr1, use_radians=False)
    r2 = RotatedRect(*rr2, use_radians=False)

    # import matplotlib
    # matplotlib.use('Agg')
    from matplotlib import pyplot
    
    from descartes import PolygonPatch

    fig = pyplot.figure(1, figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.set_xlim(-10, 300)
    ax.set_ylim(-10, 300)
    #ax.set_xlim(-100, 500)
    #ax.set_ylim(-100, 500)
    # ax.set_xlim(-500, 500)
    # ax.set_ylim(-500, 500)
    print('~~~~~~~~~r1  r1.get_contour().area', r1.get_contour().area)
    print('~~~~~~~~~r2  r2.get_contour().area', r2.get_contour().area)
    ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))  # red
    ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))  # blue

    intersection_area = r1.intersection(r2)
    if intersection_area.is_empty:
        print('~~~~~~~~~inter r1.intersection(r2).area', intersection_area)
        intersection_area = 0
    else:
        print('~~~~~~~~~inter r1.intersection(r2).area', intersection_area.area)
        ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))

        iou_ratio = r1.intersection(r2).area / (r1.get_contour().area+r2.get_contour().area-intersection_area.area)
        print('iou ', iou_ratio)

    pyplot.savefig('Check iou.png')
    pyplot.show()