import argparse
import glob
import json
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, r_box_iou, output_to_target, save_and_plot_valid_result_to_tensorboard)
from utils.torch_utils import select_device, time_synchronized

from models.models import *
#from utils.datasets import *

from utils.metrics import ConfusionMatrix, ap_per_class

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x, y, w, h, θ) format.
    Arguments:
        detections (Array[N, 7]), x, y, w, h, θ, conf, class
        labels (Array[M, 6]), class, x, y, w, h, θ
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # iou = box_iou(labels[:, 1:], detections[:, :4])
    iou = r_box_iou(labels[:,1:6], detections[:,:5], useGPU=True)
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 6]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.1,  # for NMS  0.6 default
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         save_txt=False,
         agnostic_nms=False,
         theta_format='dhxdhy',
         loss_terms='hioudhxdhy',
         tb_writer=None,
         epoch=-1,
         project=ROOT / 'runs/val',  # save to project/name
         name='exp',
         whichdataset='vehicle8cls',
         angle_modify_factor=False):
    
    num_extra_outputs = get_number_of_extra_outputs(theta_format)
    n_loss_terms = get_number_of_loss_terms(loss_terms)

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        agnostic_nms_str = '_agnosticnms' if agnostic_nms else '_'
        valdir_name_suffix = 'imgsize'+str(imgsz)+'_conf'+str(conf_thres)+'_iou'+str(iou_thres)+'_'+theta_format+'_'+loss_terms+agnostic_nms_str
        save_dir = increment_dir(Path(project) / 'val' / name, valdir_name_suffix)  # increment run
        (Path(save_dir) / 'labels' if save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir

        with open(Path(save_dir) / 'opt.yaml', 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)

        merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
        # if save_txt:
        #     out = Path('inference/output')
        #     if os.path.exists(out):
        #         shutil.rmtree(out)  # delete output folder
        #     os.makedirs(out)  # make new output folder

        # # Remove previous
        # for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
        #     os.remove(f)

        # Load model
        model = Darknet(opt.cfg, extra_outs=theta_format, num_extra_outputs=num_extra_outputs).to(device)

        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            load_darknet_weights(model, weights[0])
        imgsz = check_img_size(imgsz, s=32)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()  # tensor內元素個數

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 32, opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True, extra_outs=theta_format)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc, conf=conf_thres, iou_thres=iouv[0])
    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(opt.names)
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # loss = torch.zeros(n_loss_terms-1, device=device)  # 去掉最後的total loss
    loss = torch.zeros(n_loss_terms, device=device)  # 不去掉最後的total loss
    jdict, stats, ap, ap_class = [], [], [], []
    rand_batch = random.randint(0,len(dataloader)-1)  # 每次valid時取不同batch的圖存
    counts_of_each_cls_while_testing = {}
    for batch_i, (img, targets, paths, shapes, cv_imgs) in enumerate(tqdm(dataloader, desc=s)):
        """
        cv_imgs: torch.size = (number of image in batch, img_h, img_w, img_channels)
        targets: torch.size = (該image屬於該batch的第幾個圖,[cls,x,y,w,h,extra_1,...,extra_n])  x,y,w,h,extra_n in normalized format(extra_n 需要的話)
        img: img size -> network input size
        """
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model, -1, -1, -1, theta_format, loss_terms, num_extra_outputs, \
                                    n_loss_terms, use_angle_modify_factor=angle_modify_factor, counts_of_each_cls=counts_of_each_cls_while_testing)[1][:n_loss_terms]  # GIoU, obj, cls
                # loss += compute_loss([x.float() for x in train_out], targets, model, -1, -1, theta_format, loss_terms, num_extra_outputs, n_loss_terms)[1][:n_loss_terms-1]  # GIoU, obj, cls 不顯示total_loss

            # 在training時，每結束N個epoch會valid一次，每valid一次就從valid batch中隨機一個batch的第一張圖存到tensorboard
            if training and batch_i == rand_batch:
                save_and_plot_valid_result_to_tensorboard('eval_rbox_before_nms.jpg', inf_out, conf_thres, theta_format, num_extra_outputs, copy.deepcopy(cv_imgs[0]), save_dir, epoch, tb_writer)

            # Run NMS
            t = time_synchronized()
            # output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge, num_extra_outputs=num_extra_outputs)
            output = rotate_non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge, agnostic=agnostic_nms, \
                                                theta_format=theta_format, num_extra_outputs=num_extra_outputs, whichdataset=whichdataset)
            t1 += time_synchronized() - t

        if training and batch_i == rand_batch:
            save_and_plot_valid_result_to_tensorboard('eval_rbox_after_nms.jpg', output, conf_thres, theta_format, num_extra_outputs, copy.deepcopy(cv_imgs[0]), save_dir, epoch, tb_writer)
            
        # output shape: (num_img_in_batch, 7[x,y,w,h,θ,conf,classid])
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                # txt_path = str(out / Path(paths[si]).stem)
                txt_path = str(save_dir / Path(paths[si]).stem)
                pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in pred:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

            # Clip boxes to image bounds
            # clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = Path(paths[si]).stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                # tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                if theta_format.find('dhxdhy') != -1:
                    dx, dy = labels[:,-num_extra_outputs]-labels[:,1], labels[:,-num_extra_outputs+1]-labels[:,2]
                    theta = torch.atan2(-dy,dx)
                elif theta_format.find('sincos') != -1:
                    theta = torch.atan2(labels[:,5],labels[:,6])
                else: raise 'Unsupported train mode in test.py'

                theta = torch.where(theta<0, theta+2*math.pi, theta)
                tbox_xywhtheta = torch.cat((labels[:, 1:5]*whwh, theta.view(-1,1)), dim=1)

                confusion_matrix.process_batch(pred, torch.cat((labels[:,0:1], tbox_xywhtheta), 1))
                # correct = process_batch(pred, torch.cat((labels[:,0:1], tbox_xywhtheta), 1), iouv)
                # confusion_matrix.plot(label_content=0, save_dir=save_dir, names=names)
                # exit(0)

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 6]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # convert (x,y,w,h,theta)->(x,-y,w,h,theta-pi/2) 
                        # shapely, differentiable rotate iou都是這樣的格式
                        # 計算rotate iou前都需要經過轉換
                        tbox4nms = tbox_xywhtheta.clone()
                        tbox4nms[:,1] = -tbox_xywhtheta[:,1]
                        tbox4nms[:,4] = torch.where(tbox4nms[:,4]-math.pi/2<0, tbox4nms[:,4]-math.pi/2+2*math.pi, tbox4nms[:,4]-math.pi/2)
                        pbox4nms = pred[pi,:5].clone()
                        pbox4nms[:,1] = -pred[pi,1]
                        pbox4nms[:,4] = torch.where(pbox4nms[:,4]-math.pi/2<0, pbox4nms[:,4]-math.pi/2+2*math.pi, pbox4nms[:,4]-math.pi/2)
                        ious, i = r_box_iou(pbox4nms, tbox4nms[ti], useGPU=True).max(1)

                        # Append detections
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))

        # Plot images
        if batch_i < 1:
            # f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            f = Path(save_dir) / 'test_batch_gt.jpg'  # filename
            plot_images(img, targets, paths, str(f), names, theta_format=theta_format, num_extra_outputs=num_extra_outputs, plotontb=True, isGT=True)  # ground truth
            f = Path(save_dir) / 'test_batch_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, str(f), names, theta_format=theta_format, num_extra_outputs=num_extra_outputs, plotontb=True, isGT=False)  # predictions

    # print('----------------------')
    # print('counts_of_each_cls_while_testing',  counts_of_each_cls_while_testing)
    # print('----------------------')

    # Output each class counts per valid epoch while testing
    cls_counts_file = str(Path(save_dir) / 'testing_cls_counts.txt')
    with open(cls_counts_file, 'a') as outfile:
        counts_list = [counts_of_each_cls_while_testing[i] if i in counts_of_each_cls_while_testing else 0 for i in range(nc)]
        s = ('%8d' + '%10d' * nc + '\n') % (epoch, *counts_list)
        outfile.write(s)

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        # p, r, ap, f1, ap_class = ap_per_class(*stats)  # yolov4
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)   # yolov5
        ap50, ap = ap[:, 0], ap.mean(1)
        # p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95] #yolov4
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12d' * 2 + '%12.3g' * 4  # print format
    # pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    confusion_matrix.plot(label_content=0, save_dir=save_dir, names=names)    

    # Save results
    if not training:
        with open(Path(save_dir) / 'val_results.txt', 'w') as outfile:
            outfile.write(s+'\n')
            outfile.write(pf % ('all', seen, nt.sum(), mp, mr, map50, map)+'\n')
            for i, c in enumerate(ap_class):
                outfile.write(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i])+'\n')
            outfile.write('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')

    parser.add_argument('--theta-format', type=str, default='dhxdhy', help='traning mode')
    parser.add_argument('--loss-terms', type=str, default='hioudhxdhy', help='box regression method and extra loss terms')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--which-dataset', type=str, default='vehicle8cls', help='for nms use with same iou thres or different iou thres.')

    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             agnostic_nms=opt.agnostic_nms,
             theta_format=opt.theta_format,
             loss_terms=opt.loss_terms,
             project=opt.project,
             name=opt.name,
             whichdataset=opt.which_dataset)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(352, 832, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # plot
