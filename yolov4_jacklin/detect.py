import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
from torch._C import _cxx_flags
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, rotate_non_max_suppression, apply_classifier, scale_coords, r_scale_coords_new, xyxy2xywh, xywhtheta24xy_new, draw_one_polygon, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from models.experimental import *
from utils.datasets import *
from utils.general import *

merged_cls_dict = {'person':'0', 'bike':'1', 'moto':'2', 'sedan':'3', 'truck':'4', 'bus':'5', 'tractor':'6', 'trailer':'7'}

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names, theta_format, \
    save_img_before_nms, which_dataset, save_label_car_tool_format_txt, \
    save_vehicle8cls_IOTMOTC_format_txt, save_dota_txt = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, \
        opt.img_size, opt.cfg, opt.names, opt.theta_format, opt.save_img_before_nms, \
        opt.which_dataset, opt.save_label_car_tool_format_txt, \
        opt.save_vehicle8cls_IOTMOTC_format_txt, opt.save_dota_txt

    # dota detect clstxt directory name suffix
    weights_dir_name = weights[0][weights[0].find('exp'):].split(os.sep)[0]
    agnostic_nms_str = ''
    if opt.agnostic_nms: agnostic_nms_str = 'agnostic_nms'
    if opt.augment: augment_str = 'tta'
    else: augment_str = ''
    dota_detect_clstxt_suffix = '{}-{}-conf{}iou{}-{}-{}'.format(weights_dir_name.split('_')[0], weights_dir_name.split('_')[1][:14], opt.conf_thres, opt.iou_thres, agnostic_nms_str, augment_str)

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    num_extra_outputs = get_number_of_extra_outputs(theta_format)

    if save_dota_txt:
        dota_detect_cls_txt_dir = os.getcwd() + os.sep + 'dota_detect_clstxt-' + dota_detect_clstxt_suffix
        if os.path.exists(dota_detect_cls_txt_dir):
            shutil.rmtree(dota_detect_cls_txt_dir)  # delete output folder
        os.makedirs(dota_detect_cls_txt_dir)  # make new output folder 

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz, extra_outs=theta_format, num_extra_outputs=num_extra_outputs).cuda()
    try:
        model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    except:
        load_darknet_weights(model, weights[0])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer, vid_writer_nms = None, None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = opt.save_img_video
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    avg_inference_nms_t = 0.0
    img_count = 0

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        #----------------------
        # draw pred rbox before nms
        if save_img_before_nms:
            img_before_nms = im0s.copy()
            for x in pred:
                x = x[x[:, 4] > opt.conf_thres]  # confidence
    
                if x.shape[0] == 0: continue

                if theta_format.find('dhxdhy') != -1:
                    delta_x, delta_y = x[:,-num_extra_outputs]-x[:,0], x[:,-1]-x[:,1]
                    convert_theta = torch.atan2(-delta_y, delta_x)
                elif theta_format.find('sincos') != -1:
                    convert_theta = torch.atan2(x[:,-num_extra_outputs], x[:,-1])

                convert_theta = torch.where(convert_theta<0,convert_theta+2*math.pi,convert_theta)
                rbox_xywhtheta = torch.cat((x[:,:4], convert_theta.view(-1,1)), dim=1).cpu().numpy()
                all_pts = xywhtheta24xy_new(rbox_xywhtheta)
                # all_pts[:,:] /= gain
                
                all_pts = r_scale_coords_new(img.shape[2:], all_pts, im0s.shape)
                for idx, (pts, conf) in enumerate(zip(all_pts, x[:,4])):
                    # draw_one_rbbox(img_check_boxes, pts.cpu().numpy())
                    draw_one_polygon(img_before_nms, pts, 0)
                    cv2.putText(img_before_nms, '{:.2f}'.format(conf), (int(pts[0]), int(pts[1])), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    
                    if theta_format.find('dxdy') != -1:
                        hx, hy = (pts[0]+pts[2])/2, (pts[1]+pts[3])/2
                        cx, cy = (pts[0]+pts[4])/2, (pts[1]+pts[5])/2
                        cv2.circle(img_before_nms, (int(hx), int(hy)), 2, (0,0,255), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.circle(img_before_nms, (int(cx), int(cy)), 2, (0,255,0), thickness=-1, lineType=cv2.LINE_AA)
        #----------------------

        # Apply NMS
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, num_extra_outputs=num_extra_outputs)
        pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, \
                                            theta_format=theta_format, num_extra_outputs=num_extra_outputs, whichdataset=which_dataset)
        t2 = time_synchronized()
        
        img_count += 1
        avg_inference_nms_t += (t2 - t1)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # pred shape: (num_img_in_batch, 7[x,y,w,h,θ,conf,classid])
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')

            # Write frame ID to vehicle8cls_IOTMOTC_format_txt
            if save_vehicle8cls_IOTMOTC_format_txt:
                # assert dataset.cap, 'Only support video input while saving vehicle8cls_IOTMOTC_format_txt!!'
                if dataset.cap:
                    output_8cls_str = str(dataset.frame)
                else:
                    output_8cls_str = Path(p).name

                # with open(save_path[:save_path.rfind('.')] + '_8cls.txt', 'a') as file:
                #     file.write('{}\n'.format(output_8cls_str))
            
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # det shape (num_of_det, 10[x1,y1,x2,y2,x3,y3,x4,y4,conf,classid])
                det4pts = xywhtheta24xy_new(det[:,:5])
                det4pts = r_scale_coords(img.shape[2:], det4pts, im0.shape)
                det = torch.cat((det4pts,det[:,5:]), dim=1)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results  
                for *xyxy, conf, cls in det:  # xyxy: x1y1x2y2x3y3x4y4 clockwise
                    if save_txt:  # Write to file
                        xywh = (fourxy2xywh(torch.tensor(xyxy).view(1, 8))/gn).view(-1)  # normalized xywh
                        dx,dy = xyxy[0]-xyxy[6], xyxy[1]-xyxy[7]
                        theta = torch.atan2(-dy,dx)
                        theta = theta+2*math.pi if theta < 0 else theta
                        xywhtheta = torch.cat((xywh,torch.tensor([theta])),dim=0).tolist()
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 6 + '\n') % (cls, *xywhtheta))  # label format

                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    
                    if save_label_car_tool_format_txt:
                        xyxy = torch.tensor(xyxy)
                        xywh = (fourxy2xywh(xyxy.view(1, 8))/gn).view(-1)
                        out_theta = torch.atan2(xyxy[7]-xyxy[1], xyxy[6]-xyxy[0])
                        # label_it_car的角點順序是左上角點逆時針 x1y1x2y2x3y3x4y4 -> x1y1x4y4x3y3x2y2
                        pts_roll = torch.roll(xyxy.view(-1,2), -1, 0).flip(0).view(-1)
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            # file.write(('%g ' * 5 + '\n') % (obj_cls, *xywh))  # label format
                            file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(int(cls), *xywh, *pts_roll, out_theta))  # GUI label format

                    if save_vehicle8cls_IOTMOTC_format_txt:
                        xyxy = torch.tensor(xyxy)
                        pts_int = torch.round(xyxy).int()
                        if (pts_int<0).any() or (pts_int[[0,2,4,6]]>=im0.shape[1]).any() or (pts_int[[1,3,5,7]]>=im0.shape[0]).any():
                            pass
                        else:
                            pts_int = pts_int.tolist()
                            output_8cls_str += ' ' + merged_cls_dict[names[int(cls)]]
                            output_8cls_str += ' ' + '{:.2f}'.format(conf) + ' ' + str(pts_int[0]) + ' ' + str(pts_int[1]) \
                                                + ' ' + str(pts_int[2]) + ' ' + str(pts_int[3]) + ' ' + str(pts_int[4]) \
                                                + ' ' + str(pts_int[5]) + ' ' + str(pts_int[6]) + ' ' + str(pts_int[7])

                    if save_dota_txt:
                        img_name_wo_ext = Path(p).name.split('.')[0]
                        pts = torch.tensor(xyxy).tolist()
                        out_dota_str = '%s %.12f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (img_name_wo_ext, conf,
                                                                                             pts[0], pts[1], pts[2], pts[3],
                                                                                             pts[4], pts[5], pts[6], pts[7]
                                                                                             )
                        
                        with open(dota_detect_cls_txt_dir+os.sep+"Task1_"+names[int(cls)]+".txt", "a+") as file:
                            file.write(out_dota_str)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        tensor_xyxy = torch.tensor(xyxy)
                        # 由於上面用*xyxy從det衷取出八個xy值，會使得*xyxy以list的方式儲存
                        draw_one_polygon(im0, tensor_xyxy, int(cls), line_thickness=2)
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if save_img_before_nms:
                    extension_idx = -save_path[::-1].find('.')
                    before_nms_save_path = save_path[:extension_idx-1] + '_before_nms' + save_path[extension_idx-1:]

                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                    if save_img_before_nms: cv2.imwrite(before_nms_save_path, img_before_nms)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))     
                        if save_img_before_nms:
                            vid_writer_nms = cv2.VideoWriter(before_nms_save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

                    if save_img_before_nms:
                        vid_writer_nms.write(img_before_nms)

            if save_vehicle8cls_IOTMOTC_format_txt:
                with open(save_path[:save_path.rfind('.')] + '_8cls.txt', 'a') as file:
                    file.write('{}\n'.format(output_8cls_str)) 

    if vid_writer is not None: vid_writer.release()
    if vid_writer_nms is not None: vid_writer_nms.release()

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    print('Avg inferene+nms time. (%.3fs)' % (avg_inference_nms_t/img_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')

    parser.add_argument('--theta-format', type=str, default='dhxdhy', help='theta format to convert')
    parser.add_argument('--save-img-before-nms', action='store_true', help='update all models')
    parser.add_argument('--which-dataset', type=str, default='vehicle8cls', help='for nms use with same iou thres or different iou thres.')
    parser.add_argument('--save-label-car-tool-format_txt', action='store_true', help='save results in label_it_car format to *.txt. All results in one txt file')
    parser.add_argument('--save-vehicle8cls-IOTMOTC-format-txt', action='store_true', help='save results in IoT MOTC (運研所) format to *.txt. All results in one txt file')
    parser.add_argument('--save-dota-txt', action='store_true', help='save results in DOTA1.0 format to *.txt.')
    parser.add_argument('--save-img-video', action='store_true', help='save result to images / video')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
