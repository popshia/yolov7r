import cv2
import numpy as np
# import torch

# def draw_one_bbox(im, ):


def draw_one_rbbox(im, pts, cls=None):
    """
    Draw a rotated bounding box

    Input:
        pts: Four vertices coordinates of rotated bounding box,
             the ordered is starting from top left vertex in clockwise.

             shape: (8,) in numpy shape
                    [x1,y1,x2,y2,x3,y3,x4,y4]

    Output:
        Do not need to return the image
    """
    # if isinstance(x, torch.Tensor)
    # 若點的數值是存成浮點數，則先轉成整數
    pts_np_int = pts.astype(int)

    if cls is not None:
        cv2.line(im, (pts_np_int[0], pts_np_int[1]), (pts_np_int[2], pts_np_int[3]), (255,0,0), thickness=1, lineType=cv2.LINE_AA)
    else:
        cv2.line(im, (pts_np_int[0], pts_np_int[1]), (pts_np_int[2], pts_np_int[3]), (255,0,0), thickness=1, lineType=cv2.LINE_AA)

    # print('pts_np_int before expand dim', pts_np_int)
    # print('pts_np_int before expand dim shape', pts_np_int.shape)

    # 純顯示需求，用cv2.polylines畫框，為了將框開口朝車頭，所以得先將點1跟點4順序做調換
    # 確保是在cpu模式下計算才能將pts維度提昇，並置換點1點4座標的column
    pts2draw = np.expand_dims(pts_np_int, axis=0)
    pts2draw = np.concatenate((pts2draw[:,2:], pts2draw[:,0:2]),axis=1)
    # pts2draw = torch.index_select(pts.cpu().unsqueeze(0), 1, torch.LongTensor([2,3,4,5,6,7,0,1]))
    # pts2draw = torch.squeeze(pts2draw)

    # print('pts2draw after expand dim', pts2draw)
    # print('pts2draw after expand dim shape', pts2draw.shape)
    
    # 將pts2draw的格式轉成cv2.polylines接受的格式 numpy int [n(點的數量),2]
    cv2.polylines(im, [pts2draw.reshape(-1,2)], isClosed=False, color=(0,255,255), thickness=1, lineType=cv2.LINE_AA)
    # cv2.polylines(img_check_boxes, [pts.cpu().numpy().astype(int).reshape(-1,2)], isClosed=True, color=(0,255,255), thickness=2, lineType=cv2.LINE_AA)
    # cv2.line(img_check_boxes, (pts[0], pts[1]), (pts[2], pts[3]), (0,255,255), thickness=2, lineType=cv2.LINE_AA)
    # cv2.line(img_check_boxes, (pts[2], pts[3]), (pts[4], pts[5]), (0,255,255), thickness=2, lineType=cv2.LINE_AA)
    # cv2.line(img_check_boxes, (pts[4], pts[5]), (pts[6], pts[7]), (0,255,255), thickness=2, lineType=cv2.LINE_AA)
    # cv2.line(img_check_boxes, (pts[6], pts[7]), (pts[0], pts[1]), (0,255,255), thickness=2, lineType=cv2.LINE_AA)
    pass