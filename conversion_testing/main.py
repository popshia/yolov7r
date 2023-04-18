import cv2
import os
import argparse
import torch
from general import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="conversion testing")
    parser.add_argument("--multiple", action="store_true", help="flag for testing multiple label conversion.")
    args = parser.parse_args()
    label_directory = os.path.join(os.getcwd(), "labels")
    image_directory = os.path.join(os.getcwd(), "images")

    if args.multiple:
        label_file = os.path.join(label_directory, "four.txt")
        img_file = os.path.join(image_directory, "four.bmp")
    else:
        label_file = os.path.join(label_directory, "one.txt")
        img_file = os.path.join(image_directory, "one.bmp")

    label = open(label_file)
    img = cv2.imread(img_file)

    for i, line in enumerate(label):
        variables = torch.Tensor([[float(v) for v in line.split()[1:]]])

        if i == 0:
            label_list = variables
        else:
            label_list = torch.cat((label_list, variables), dim=0)

    # print(multiple_xywhrad2poly(img.shape[1], img.shape[2], label[:, 1:5], label[:, 5]))
    # print(label_list[:, 0:5])
    point1 = multiple_xywhrad2poly(img.shape[1], img.shape[0], label_list[:, 0:4], label_list[:, 4])[0]
    print([point1[1], point1[3]])
    print([point1[1, 0], point1[3, 0]])
    print([point1[1, 1], point1[3, 1]])
    # print(multiple_xywhrad2poly(img.shape[1], img.shape[0], label_list[:, 0:4], label_list[:, 4]))