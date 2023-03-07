import glob
import math
import os
import random
import shutil
import threading
import time
from pathlib import Path
from threading import Thread
from typing_extensions import Concatenate

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy, fourxy2xywh, xywhtheta24xy_new, \
    torch_distributed_zero_first, get_number_of_extra_outputs
from utils.extractRect import extract_rect_from_poly    

help_url = ''
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      local_rank=-1, world_size=1, extra_outs='dhxdhy', image_weights=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(local_rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      image_weights=image_weights,
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      extra_outs=extra_outs)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if local_rank != -1 else None
    # https://blog.csdn.net/tsq292978891/article/details/80454568  pin_memory
    # set pin_memory=False to remove [W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             sampler=train_sampler,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class LoadImages:  # for inference
    def __init__(self, path, img_size=640):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=640):
        self.img_size = img_size

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(0 if s == '0' else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, extra_outs='dhxdhy'):
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))  # os-agnostic
                parent = str(Path(p).parent) + os.sep
                if os.path.isfile(p):  # file
                    with open(p, 'r') as t:
                        t = t.read().splitlines()
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                elif os.path.isdir(p):  # folder
                    f += glob.iglob(p + os.sep + '*.*')
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted(
                [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats])
        except Exception as e:
            raise Exception('Error loading data from %s: %s\nSee %s' % (path, e, help_url))
        
        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.extra_outs = extra_outs
        self.n_extra_outputs = get_number_of_extra_outputs(self.extra_outs)
        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride

        # Define labels
        self.label_files = [x.replace('images', 'rlabels').replace(os.path.splitext(x)[-1], '.txt') for x in
                            self.img_files]

        # Check cache
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            l = self.labels[i]  # label
            if l.shape[0]:
                assert l.shape[1] == 6, '<6 or > 6 label columns: %s' % file  # (cls,x,y,w,h,Θ)
                assert (l >= 0).all(), 'negative labels: %s' % file
                # check x,y,w,h normalized
                assert (l[:, 1:-1] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:-1] * [w, h, w, h]  # box
                        b[2:-1] = b[2:-1].max()  # rectangle to square
                        b[2:-1] = b[2:-1] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = None
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x


    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        
        if self.image_weights:
            # print('self.indices',self.indices)
            index = self.indices[index]
        
        hyp = self.hyp

        if self.mosaic:
            # Load mosaic
            # img: size=(img-size,img-size,3)   img-size you set in command
            # labels: size=(單張img4中的目標GT數量, [classid, TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_X, BL_y, Θ])
            # 四個角點的座標也已經還原成圖像上的座標系大小 角點座標未正規化(以徑度表示0-2pi)
            # 此處角點並非符合物體旋轉框的角點，而是純粹能代表物體長寬的水平框角點，
            # 如果經過旋轉的資料擴增，原本水平框的左上右下點也會跟著旋轉，
            # 這時候的左上右下點就不再能夠表示框的長寬，所以才要回傳四個角點
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            # h0: original image height, h: network input height
            # 將原圖長寬等比例縮放到設定的network input size，
            # 比例為network size/max(原圖w, 原圖h)
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                # labels: size=(單張img中的目標GT數量, [classid, TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_X, BL_y, Θ])
                img, labels = r_random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
                # img, labels = random_perspective(img, labels,
                #                                  degrees=hyp['degrees'],
                #                                  translate=hyp['translate'],
                #                                  scale=hyp['scale'],
                #                                  shear=hyp['shear'],
                #                                  perspective=hyp['perspective'])


            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        # 將labels轉回: (cls,x,y,w,h,Θ)  並對xywh normalize
        if nL:
            # 如果有做資料擴增，回傳的labels就會變成下面格式
            # (cls,TL_x,TL_y,TR_x,TR_y,BR_x,BR_y,BL_x,BL_y,Θ) 座標已還原成未normalize大小
            if self.augment:
                labels_xywh = fourxy2xywh(labels[:,1:9])
                labels = np.concatenate((labels[:,0].reshape(-1,1),labels_xywh,labels[:,-1].reshape(-1,1)), axis=1)
            else:
            # 沒有做資料擴增，labels的格式還是[cls,TL_x,TL_y,BR_x,BR_y,0]    
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh

            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1
        else:
            # (cls,x,y,w,h,Θ)
            if isinstance(labels, np.ndarray):
                labels = np.empty((0,6),dtype=np.float64)
            if isinstance(labels, torch.Tensor):
                labels = torch.empty((0,6), dtype=torch.float64)
                print('!!!!!!!!!! become tensor !!!!!!!!!!!!')
                exit(0)

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]  # y變x不變
                    labels[:,-1] = np.where(labels[:,-1] != 0, 2*math.pi-labels[:,-1], 0.)  # 角度跟著改變 0度上下翻轉後還是要設為0度

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]  # x變y不變
                    labels[:,-1] = np.where(labels[:,-1] > math.pi, 3*math.pi-labels[:,-1], math.pi-labels[:,-1])  # 角度跟著改變

        # check_labels = labels.copy()
        # check_label_img = img.copy()
        # check_labels[:,[1,3]]*=img.shape[1]
        # check_labels[:,[2,4]]*=img.shape[0]
        # rbox_xywhtheta = np.concatenate((check_labels[:,1:5],check_labels[:,-1].reshape(-1,1)),axis=1)
        # rbox4pts = xywhtheta24xy_new(rbox_xywhtheta)
        # print('self.extra_outs', self.extra_outs)
        # for pts in rbox4pts:
        #     cv2.circle(check_label_img, (int(pts[0]),int(pts[1])), 3, (255,0,0), -1, cv2.LINE_AA)
        #     cv2.circle(check_label_img, (int(pts[2]),int(pts[3])), 3, (255,0,0), -1, cv2.LINE_AA)
        #     cv2.circle(check_label_img, (int(pts[4]),int(pts[5])), 3, (255,0,0), -1, cv2.LINE_AA)
        #     cv2.circle(check_label_img, (int(pts[6]),int(pts[7])), 3, (255,0,0), -1, cv2.LINE_AA)
        # cv2.imshow('img check label',check_label_img)
        # cv2.waitKey(0)
        
        # 要取label之前先按照extra_outs轉換成需要的格式
        # labels: (cls,x,y,w,h,Θ) -> e.g.,(cls,x,y,w,h,sin,cos)
        labels = convert_target_format(labels, self.extra_outs)

        # 初始化標籤框對應的圖片序號，配合下面的collate_fn使用
        # labels_out: (img_id,cls,x,y,w,h,sin,cos)
        labels_out = torch.zeros((nL, labels.shape[1]+1))

        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        cv_img = img.copy()
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416 (CHW)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, cv_img

    @staticmethod
    def collate_fn(batch):
        """
        n img in batch
        img: tuple size n (tensor 1, ... , tensor n)  tensor 1 size [3, imgh, imgw]
        label: tuple size n (tensor 1, ... , tensor n)  tensor 1 size [num of targets, [img_id,cls,x,y,w,h,Θ]]
        shapes: tuple size n (tensor 1, ... , tensor n)
        cv_img: tuple size n (ndarray 1, ... , ndarray n)  ndarray 1 size [imgh,imgw,3]
        """
        img, label, path, shapes, cv_img = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, np.asarray(cv_img)


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    yc, xc = s, s  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:-1], 0, 2 * s, out=labels4[:, 1:-1])  # use with random_affine

        # Replicate
        # img4, labels4 = replicate(img4, labels4)

    # draw horizontal TL and BR point
    # img4_show = img4.copy()
    # for label in labels4:
    #     cv2.circle(img4_show, (int(label[1]), int(label[2])), 5, (255,0,0), 2, cv2.LINE_AA)
    #     cv2.circle(img4_show, (int(label[3]), int(label[4])), 5, (0,0,255), 2, cv2.LINE_AA)

    # img4_rz = cv2.resize(img4_show, (800, 800), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('img before augmentation', img4_rz)
    # cv2.waitKey(0)

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    # after random perspective labels4 size=[num_keep_targets, 10(cls,tlx,tly,trx,try,blx,bly,brx,bry,Θ)]
    img4, labels4 = r_random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove
    # img4, labels4 = random_perspective(img4, labels4,
    #                                    degrees=self.hyp['degrees'],
    #                                    translate=self.hyp['translate'],
    #                                    scale=self.hyp['scale'],
    #                                    shear=self.hyp['shear'],
    #                                    perspective=self.hyp['perspective'],
    #                                    border=self.mosaic_border)  # border to remove

    # imgshow=img4.copy()
    # for l in labels4:
    #     cv2.circle(imgshow, (int(l[1]), int(l[2])), 5, (255,0,0), -1, cv2.LINE_AA)
    #     cv2.circle(imgshow, (int(l[3]), int(l[4])), 5, (0,255,0), -1, cv2.LINE_AA)
    #     cv2.circle(imgshow, (int(l[5]), int(l[6])), 5, (0,0,255), -1, cv2.LINE_AA)
    #     cv2.circle(imgshow, (int(l[7]), int(l[8])), 5, (120,120,255), -1, cv2.LINE_AA)
    # xy = (labels4[:,[1,2]]+labels4[:,[5,6]])/2
    # wh = np.sum(((labels4[:,[3,4,5,6]]-labels4[:,[1,2,3,4]])**2).reshape(-1,2,2),axis=1)**0.5
    # xywhtheta = np.concatenate((xy,wh,labels4[:,-1].reshape(-1,1)),axis=1)
    # rbox4pts = xywhtheta24xy_new(xywhtheta)
    # for l in rbox4pts:
    #     cv2.circle(imgshow, (int(l[0]), int(l[1])), 5, (255,0,0), -1, cv2.LINE_AA)
    #     cv2.circle(imgshow, (int(l[2]), int(l[3])), 5, (0,255,0), -1, cv2.LINE_AA)
    #     cv2.circle(imgshow, (int(l[4]), int(l[5])), 5, (0,0,255), -1, cv2.LINE_AA)
    #     cv2.circle(imgshow, (int(l[6]), int(l[7])), 5, (120,120,255), -1, cv2.LINE_AA)
    # imgshow = cv2.resize(imgshow, (1024,1024), interpolation=cv2.INTER_AREA)
    # cv2.imshow('mosaic image', imgshow)
    # cv2.waitKey(0)

    return img4, labels4


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def r_random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """
    隨機旋轉，平移，縮放，推移錯切
    Input:
        img: shape=(height,width,3)
        targets: size=(單張圖片中的目標數量,[class,TL_x,TL_y,BR_x,BR_y,Θ])
                此處的左上右下點是水平框的，不是原本旋轉框的
    Output:
        img: shape=(height,width,3)
        new_targets: (目標數量,[cls,x1,y1,x2,y2,x3,y3,x4,y4,Θ])  角點左上角(x1,y1)起頭順時針排序

        四個角點的座標也已經還原成圖像上的座標系大小 角點座標未正規化(以徑度表示0-2pi)，
        此處角點並非符合物體旋轉框的角點，而是純粹能代表物體長寬的水平框角點，
        如果經過旋轉的資料擴增，原本水平框的左上右下點也會跟著旋轉，
        這時候的左上右下點就不再能夠表示框的長寬，所以才要回傳四個角點
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # random.uniform(x,y)    x <= N <= y

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    # print('border', border)
    # print('M', M)
    # print('np.eye(3)', np.eye(3))
    # print('(M != np.eye(3)).any()', (M != np.eye(3)).any())
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # cv2.imshow('img after warp', img)
    # cv2.waitKey(0)

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        """
        如果只給左上右下點，圖片沒有旋轉的情況下框的長寬不會錯，因為左上右下就能計算出來長寬
        但是如果圖片經過旋轉，旋轉過後的左上右下就沒辦法表達原本的長寬，
        所以這邊取四個角點
        """
        # 此時的框如果augmentation有旋轉會變成旋轉框，否則還是正框，
        # 將框的四個角點重新排序左上角為第一個點順時針
        clockwise_xy = np.concatenate((xy[:,[0,1]], xy[:,[6,7]], xy[:,[2,3]], xy[:,[4,5]]),axis=1)

        # (cls,x1,y1,x2,y2,x3,y3,x4,y4,Θ)
        targets = np.concatenate((targets[:,0].reshape(-1,1), clockwise_xy, targets[:,-1].reshape(-1,1)),axis=1)

        # 如果有物體而且label有10個元素(cls,x1,y1,x2,y2,x3,y3,x4,y4,Θ)
        if targets.shape[0] and targets.shape[1] == 10:
            # 資料擴增如果有角度旋轉，則調整角度
            targets[:,-1] += a*math.pi/180
            targets[:,-1] = np.where(targets[:,-1]<0, 2*math.pi+targets[:,-1], np.where(targets[:,-1]>=2*math.pi, targets[:,-1]-2*math.pi, targets[:,-1]))

        """
        剩下的augmentation不會再讓框跑出畫面(HSV,flip)，所以在這邊刪不要的框
        四個點都在外面 or 頭在外面 or 中心在外面
        """
        # 此時不能隨便clip框，因為不管現在的框是旋轉或是正框，如果直接clip會直接縮短框長邊或寬邊的長度，
        # 這樣經過旋轉過後的框對回去原本物體，長寬會對不上
        cand_idx = r_box_candidates(box1=targets, imgw=img.shape[1], imgh=img.shape[0])
        targets = targets[cand_idx]

        """
        https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB/issues/70
        暫時不對超出邊界的框做處理 讓網路學到完整框大小

        最後再處理符合上述條件的框(框頭點在影像內,框中心點在影像內,框的至少一個角點在影像內)
        將超出邊界的部份裁切掉,問題可轉化為 找出凸多邊形內最大矩形框
        """
        # # 找出多邊形內最大的矩形: extract_rect_from_poly函數的input需要binary的polygon圖，所以先做以下處理
        # # 1.將intersection_point_list轉成numpy format，為了準確度以及符合convexHull函式能用的input，dtype設為np.float32
        # poly_pts = np.array(intersection_point_list, dtype=np.float32)
        # # 2.由於上面找完的poly點並沒有順時針排序，於是使用convexHull函式來找出順序，可以這樣做的原因是我們的polygon都是凸多邊形
        # # 經過squeeze函式並轉成int32是為了符合fillPoly函式
        # hull = cv2.convexHull(poly_pts).squeeze().astype('int32')
        # # 3.填充之前先用polylines函式，可以使填充區域邊緣更光滑，肉眼看感覺差不多阿@@
        # cv2.polylines(im, pts=[hull], isClosed=True, color=(255), thickness=2, lineType=cv2.LINE_AA)
        # # 4.產生polygon的binary圖
        # cv2.fillPoly(im, pts=[hull], color=(255), lineType=cv2.LINE_AA)
        # rect_pts_list, angle = extract_rect_from_poly(im)
        


        # # create new boxes
        # x = xy[:, [0, 2, 4, 6]]
        # y = xy[:, [1, 3, 5, 7]]
        # xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # # apply angle-based reduction of bounding boxes
        # # radians = a * math.pi / 180
        # # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # # x = (xy[:, 2] + xy[:, 0]) / 2
        # # y = (xy[:, 3] + xy[:, 1]) / 2
        # # w = (xy[:, 2] - xy[:, 0]) * reduction
        # # h = (xy[:, 3] - xy[:, 1]) * reduction
        # # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # # clip boxes
        # xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        # xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # # filter candidates
        # i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        # targets = targets[i]
        # targets[:, 1:5] = xy[i]

        # img_sw = img.copy()
        # for pts in targets:
        #     cv2.circle(img_sw, (int(pts[1]), int(pts[2])), 5, (255,0,0), -1, cv2.LINE_AA)
        #     cv2.circle(img_sw, (int(pts[2]), int(pts[3])), 5, (0,255,0), -1, cv2.LINE_AA)
        #     cv2.circle(img_sw, (int(pts[4]), int(pts[5])), 5, (0,0,255), -1, cv2.LINE_AA)
        #     cv2.circle(img_sw, (int(pts[6]), int(pts[7])), 5, (120,120,255), -1, cv2.LINE_AA)

        # img_rz = cv2.resize(img_sw, (900,900), interpolation=cv2.INTER_AREA)
        # cv2.imshow('img after affine', img_rz)
        # cv2.waitKey(0)
        # exit(0)

    return img, targets


def r_box_candidates(box1, box2=None, wh_thr=2, ar_thr=20, area_thr=0.1, imgw=640, imgh=640):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    # w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    # w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    # ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio

    # return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

    # 目標：透過new_targets找出頭中點
    # 先將正框/旋轉框四個角點[這個框並沒有跟物體對齊，但可以推出物體的中心點跟長寬資訊]以及角度
    # 合併推導出成xywhtheta來計算真正符合物體旋轉框的四個角點，以此找出頭中心點
    xy = (box1[:,[1,2]]+box1[:,[5,6]])/2
    wh = ((box1[:,[3,5]]-box1[:,[1,3]])**2+(box1[:,[4,6]]-box1[:,[2,4]])**2)**0.5 
    rboxs_xywhtheta = np.concatenate((xy,wh,box1[:,9].reshape(-1,1)), axis=1)
    # print('in datasets.py r_box_candidates rboxs_xywhtheta dtype', rboxs_xywhtheta.dtype)
    rbox4pts = xywhtheta24xy_new(rboxs_xywhtheta)
    hxy = (rbox4pts[:,[0,1]]+rbox4pts[:,[2,3]])/2

    # 頭中心點在影像內的遮罩
    hxy_in_img_mask = (0<=hxy[:,0])&(hxy[:,0]<imgw)&(0<=hxy[:,1])&(hxy[:,1]<imgh)

    # 四個角點都在影像外的遮罩
    corners_out_img_mask = (((rbox4pts[:,0::2]<0)|(rbox4pts[:,0::2]>=imgw)) & ((rbox4pts[:,1::2]<0) | (rbox4pts[:,1::2]>=imgh)))[:].all(1)

    # print('hxy_in_img_mask & ~corners_out_img_mask', sum(hxy_in_img_mask & ~corners_out_img_mask))

    # 物體中心點在影像內的遮罩
    cxy_in_img_mask = (0<=xy[:,0])&(xy[:,0]<imgw)&(0<=xy[:,1])&(xy[:,1]<imgh)

    # print('hxy_in_img_mask & ~corners_out_img_mask & cxy_in_img_mask', sum(hxy_in_img_mask & ~corners_out_img_mask & cxy_in_img_mask))

    return hxy_in_img_mask & ~corners_out_img_mask & cxy_in_img_mask


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(path='path/images', img_size=1024):  # from utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)


def recursive_dataset2bmp(dataset='path/dataset_bmp'):  # from utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + '/' + file
            s = Path(file).suffix
            if s == '.txt':  # replace text
                with open(p, 'r') as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, '.bmp')
                with open(p, 'w') as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, '.bmp'), cv2.imread(p))
                if s != '.bmp':
                    os.system("rm '%s'" % p)


def imagelist2folder(path='path/images.txt'):  # from utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def convert_target_format(targets, extra_outs):
    """
    xywhtheta to other format

    xywh
    theta in diameter
    """

    if len(targets):
        if extra_outs.find('sincos') != -1:
            # convert theta to sin_theta cos_theta
            return np.concatenate((targets[:,:-1],np.sin(targets[:,-1]).reshape(-1,1),np.cos(targets[:,-1]).reshape(-1,1)),axis=1)
        elif extra_outs.find('dhxdhy') != -1:
            # convert theta to headx heady
            # print('in datasets.py convert_target_format targets dtype', targets.dtype)
            rbox4pts = xywhtheta24xy_new(targets[:,1:])
            hxy = (rbox4pts[:,[0,1]]+rbox4pts[:,[2,3]])/2
            return np.concatenate((targets[:,:-1],hxy),axis=1)
        elif extra_outs.find('csl') != -1:
            # convert theta to degree
            targets[:,-1] = torch.round(targets[:,-1]*180/math.pi)
            if targets[:,-1] == 360: targets[:,-1] = 0
            return targets
        elif extra_outs.find('theta') != -1:
            return targets
        else:
            print('No supported train mode!')
            exit(0)
    else:
        if extra_outs.find('sincos') != -1 or extra_outs.find('dhxdhy') != -1:
            return np.empty((0,7),dtype=np.float64)
        elif extra_outs.find('csl') != -1:
            return targets
        else:
            print('No supported train mode!')
            exit(0)

# def get_number_of_extra_outputs(train_mode):

#     if train_mode.find('sincos') != -1 or train_mode.find('dxdy') != -1:
#         return 2
#     elif train_mode.find('cls') != -1:
#         return 1
#     else:
#         print('No supported train mode!')
#         exit(0)
