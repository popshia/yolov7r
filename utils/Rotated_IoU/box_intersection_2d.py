'''
torch implementation of 2d oriented box intersection

author: lanxiao li
2020.8
'''
import torch
if __package__ is None or __package__ == '':
    from cuda_op.cuda_ext import sort_v    
else:
    from .cuda_op.cuda_ext import sort_v
EPSILON = 1e-8    # default 1e-8 1e-14

check_use_inters = None
check_use_mask_inter = None
check_use_c12 = None
check_use_c21 = None
check_use_vertices = None
check_use_mask = None
check_use_sorted_indices = None

check_use_num_valid = None
check_use_mean = None
check_use_vertices_normalized = None

def box_intersection_th(corners1:torch.Tensor, corners2:torch.Tensor):
    """find intersection points of rectangles
    Convention: if two edges are collinear, there is no intersection point

    Args:
        corners1 (torch.Tensor): B, N, 4, 2
        corners2 (torch.Tensor): B, N, 4, 2

    Returns:
        intersectons (torch.Tensor): B, N, 4, 4, 2
        mask (torch.Tensor) : B, N, 4, 4; bool
    """
    # build edges from corners
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3) # B, N, 4, 4: Batch, Box, edge, point  ori code
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3) # ori code
    # line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3).double() # B, N, 4, 4: Batch, Box, edge, point
    # line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3).double()

    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3).repeat([1,1,1,4,1])
    line2_ext = line2.unsqueeze(2).repeat([1,1,4,1,1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    # zero case
    # num_mask = (num != .0)
    # t = torch.ones_like(den_t)*-1
    # t[num_mask] = den_t[num_mask] / num[num_mask]
    t = den_t / num
    # if ((den_t==0.) & (num==0.)).any():
    #     print('den_t num are all zeros')
    #     input('det ==0 pause')
    t[num == .0] = -1.
    # mask_t = (t >= 0) * (t <= 1)
    mask_t = (t > 0) * (t < 1)                # intersection on line segment 1  ori code
    # if torch.isnan(num).any() or torch.isnan(den_t).any():
    #     print('corners1', corners1)
    #     print('corners2',corners2)
    #     print('line1', line1)
    #     print('line2',line2)
    #     print('den_t', den_t)
    #     print('num', num)
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
    # zero case
    # u = torch.ones_like(den_u)*-1
    # u[num_mask] = -den_u[num_mask] / num[num_mask]
    u = -den_u / num
    u[num == .0] = -1.
    # mask_u = (u >= 0) * (u <= 1)
    mask_u = (u > 0) * (u < 1)                # intersection on line segment 2  ori code
    mask = mask_t * mask_u
    t = den_t / (num + EPSILON)                 # overwrite with EPSILON. otherwise numerically unstable
    intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)

    return intersections, mask

def box1_in_box2(corners1:torch.Tensor, corners2:torch.Tensor):
    """check if corners of box1 lie in box2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool
    """
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    ab = b - a                  # (B, N, 1, 2)
    am = corners1 - a           # (B, N, 4, 2)
    ad = d - a                  # (B, N, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)       # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)    # (B, N, 1)
    p_ad = torch.sum(ad * am, dim=-1)       # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)    # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > - 1e-6) * (p_ab / norm_ab < 1 + 1e-6)   # (B, N, 4)
    cond2 = (p_ad / norm_ad > - 1e-6) * (p_ad / norm_ad < 1 + 1e-6)   # (B, N, 4)
    return cond1*cond2

def box_in_box_th(corners1:torch.Tensor, corners2:torch.Tensor):
    """check if corners of two boxes lie in each other

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1

def build_vertices(corners1:torch.Tensor, corners2:torch.Tensor, 
                c1_in_2:torch.Tensor, c2_in_1:torch.Tensor, 
                inters:torch.Tensor, mask_inter:torch.Tensor):
    """find vertices of intersection area

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (B, N, 4)
        c2_in_1 (torch.Tensor): Bool, (B, N, 4)
        inters (torch.Tensor): (B, N, 4, 4, 2)
        mask_inter (torch.Tensor): (B, N, 4, 4)
    
    Returns:
        vertices (torch.Tensor): (B, N, 24, 2) vertices of intersection area. only some elements are valid
        mask (torch.Tensor): (B, N, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0). 
    # can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    vertices = torch.cat([corners1, corners2, inters.view([B, N, -1, 2])], dim=2) # (B, N, 4+4+16, 2)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([B, N, -1])], dim=2) # Bool (B, N, 4+4+16)
    return vertices, mask

def sort_indices(vertices:torch.Tensor, mask:torch.Tensor):
    """[summary]

    Args:
        vertices (torch.Tensor): float (B, N, 24, 2)
        mask (torch.Tensor): bool (B, N, 24)

    Returns:
        sorted_index: bool (B, N, 9)
    
    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X) 
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    num_valid = torch.sum(mask.int(), dim=2).int()      # (B, N)
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean       # normalization makes sorting easier
    global check_use_num_valid
    global check_use_mean
    global check_use_vertices_normalized
    check_use_num_valid = num_valid
    check_use_mean = mean
    check_use_vertices_normalized = vertices_normalized
    return sort_v(vertices_normalized, mask, num_valid).long()

def calculate_area(idx_sorted:torch.Tensor, vertices:torch.Tensor):
    """calculate area of intersection

    Args:
        idx_sorted (torch.Tensor): (B, N, 9)
        vertices (torch.Tensor): (B, N, 24, 2)
    
    return:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1,1,1,2])
    selected = torch.gather(vertices, 2, idx_ext)
    # for v in vertices[0,0]:
    #     print('v',v)
    # for idx in idx_ext[0,0]:
    #     print('idx',idx)
    # print('vertices.shape',vertices.shape)
    # print('idx_ext.shape',idx_ext.shape)
    # print('selected[selected.isnan()]',selected[selected.isnan()])

    total = selected[:, :, 0:-1, 0]*selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1]*selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    # print('total[total.isnan()]',total[total.isnan()])
    # print('area[area.isnan()]',area[area.isnan()])
    return area, selected

def oriented_box_intersection_2d(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate intersection area of 2d rectangles 

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    # inters shape [1, 5021, 4, 4, 2]
    # mask_inter shape [1, 5021, 4, 4]
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    """sorted_indices 可能出現問題"""
    sorted_indices = sort_indices(vertices, mask)

    global check_use_inters
    global check_use_mask_inter
    global check_use_c12
    global check_use_c21
    global check_use_vertices
    global check_use_mask
    global check_use_sorted_indices

    check_use_inters = inters
    check_use_mask_inter = mask_inter
    check_use_c12 = c12
    check_use_c21 = c21
    check_use_vertices = vertices
    check_use_mask = mask
    check_use_sorted_indices = sorted_indices

    # print('inters', inters)
    # print('mask_inter', mask_inter)
    # print('c12', c12)
    # print('c21', c21)
    # print('vertices', vertices)
    # print('mask', mask)
    # print('sorted_indices', sorted_indices)


    # print('inters shape', inters.shape)
    # print('mask_inter shape', mask_inter.shape)
    # print('c12 shape', c12.shape)
    # print('c21 shape', c21.shape)
    # print('vertices shape', vertices.shape)
    # print('mask.shape', mask.shape)
    # print('sorted_indices.shape', sorted_indices.shape)
    # exit(0)

    return calculate_area(sorted_indices, vertices)
