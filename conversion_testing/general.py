import torch
import numpy as np
import math


def single_xyxy2poly(x, rad):
	# Convert box from [x1, y1, x2, y2, rad] to [x1, y1, x2, y2, ..., x4, y4]
	# get center, w, h, rad
	center = torch.Tensor([(x[0]+x[2])/2,(x[1]+x[3])/2]) 
	w = torch.Tensor([x[2]-x[0]])
	h = torch.Tensor([x[3]-x[1]])
	rad = torch.Tensor([rad*math.pi*2+math.pi/2])
	# calculate cos and sin
	cos, sin = torch.cos(rad), torch.sin(rad)
	vector1 = torch.Tensor([(w/2 * cos, -w/2 * sin)])
	vector2 = torch.Tensor([(-h/2 * sin, -h/2 * cos)])
	point1 = center + vector1 + vector2
	point2 = center + vector1 - vector2
	point3 = center - vector1 - vector2
	point4 = center - vector1 + vector2
	return np.array([(point1[0,0], point1[0,1]),
					(point2[0,0], point2[0,1]),
					(point3[0,0], point3[0,1]),
					(point4[0,0], point4[0,1])]).astype(int)


def single_xywhrad2poly(width, height, box, rad):
    # clone box and radina
    clone_box = box.clone().detach().cpu().float()
    clone_rad = rad.clone().detach().cpu().float()

    # get center, w, h, rad
    center, w, h, rad = clone_box[:2], clone_box[2], clone_box[3], clone_rad*math.pi*2+math.pi/2

    # denormalized center and w, h
    center[0] *= width
    center[1] *= height
    w *= width
    h *= height

    # calculate cos and sin to get perpendicular vectors
    cos, sin = torch.cos(rad), torch.sin(rad)
    vector1 = torch.Tensor([(w/2 * cos, -w/2 * sin)])
    vector2 = torch.Tensor([(-h/2 * sin, -h/2 * cos)])

    # calculate four new points
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2

    return np.array([(point1[0,0], point1[0,1]),
    				 (point2[0,0], point2[0,1]),
    				 (point3[0,0], point3[0,1]),
    				 (point4[0,0], point4[0,1])]).astype(int)


def multiple_xywhrad2poly(width, height, box, rad):
    # output = np.empty([2, len(box)])

    # clone box and radina
    clone_box = box.clone().detach().cpu().float()
    clone_rad = rad.clone().detach().cpu().float()

    # get center, w, h, rad
    center, w, h, rad = clone_box[:, :2], clone_box[:, 2], clone_box[:, 3], clone_rad*math.pi*2+math.pi/2

    # denormalized center and w, h
    center[:, 0] *= width
    center[:, 1] *= height
    w *= width
    h *= height

    # calculate cos and sin to get perpendicular vectors

    cos, sin = torch.cos(rad[0]), torch.sin(rad[0])
    vector1 = torch.Tensor([(w[0]/2 * cos, -w[0]/2 * sin)])
    vector2 = torch.Tensor([(-h[0]/2 * sin, -h[0]/2 * cos)])

    point1 = center[0] + vector1 + vector2
    point2 = center[0] + vector1 - vector2
    point3 = center[0] - vector1 - vector2
    point4 = center[0] - vector1 + vector2
    # print(point1, point2, point3, point4)

    output = np.array(np.array([[(point1[0, 0], point1[0, 1]),
                                 (point2[0, 0], point2[0, 1]),
    				             (point3[0, 0], point3[0, 1]),
    				             (point4[0, 0], point4[0, 1])]]).astype(int))

    # calculate four new points and append
    for i in range(1, len(box)):
        cos, sin = torch.cos(rad[i]), torch.sin(rad[i])
        vector1 = torch.Tensor([(w[i]/2 * cos, -w[i]/2 * sin)])
        vector2 = torch.Tensor([(-h[i]/2 * sin, -h[i]/2 * cos)])

        point1 = center[i] + vector1 + vector2
        point2 = center[i] + vector1 - vector2
        point3 = center[i] - vector1 - vector2
        point4 = center[i] - vector1 + vector2

        output = np.append(output,
                           np.array([[(point1[0, 0], point1[0, 1]),
    				                  (point2[0, 0], point2[0, 1]),
    				                  (point3[0, 0], point3[0, 1]),
    				                  (point4[0, 0], point4[0, 1])]]).astype(int),
                           axis=0)
    return output


def xywhrad2poly(x):
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

    # 中心點到任一角點的距離，因為是矩形到四個角點距離相等
    r = (x[:,2]*x[:,2]+x[:,3]*x[:,3])**0.5/2.0
    # 中心點到第一點的向量與中心點到頭中點的向量角度會小於90度所以不用再判斷值是否小於0

    # diff_radian = my_atan2((x[:,2]/2),(x[:,3]/2))
    diff_radian = x[:, 4]*math.pi*2+math.pi/2

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

    if isinstance(x, torch.Tensor):
        return torch.stack((x1,y1,x2,y2,x3,y3,x4,y4), dim=0).T
    elif isinstance(x, np.ndarray):
        return np.stack((x1,y1,x2,y2,x3,y3,x4,y4), axis=0).T
    else:
        print('No supported type')
        exit(0)


def poly2xywh(poly):
    xy = (poly[:, [0, 1]]+poly[:, [4, 5]])/2
    wh = ((poly[:, [2, 4]]-poly[:, [0, 2]])**2+(poly[:, [3, 5]]-poly[:, [1, 3]])**2)**0.5
    return np.concatenate((xy, wh), axis=1)

