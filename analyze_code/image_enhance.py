# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:38:13 2019
使用imgaug做在线数据增强（需要找到主程序数据传入口。即有cv2.imread或别的读入信息）
返回的即使cv2可以直接使用的图片格式
低下有使用cv2做的数据增强
"""

# import tensorflow as tf
import cv2
import random
import os

# import numpy as np
index = random.randint(1, 100)
#
# file_name = os.listdir('./tem/')
# out_dir = './out_img/'
# base_dir = './tem/'

file_name = os.listdir('./label_img/')
out_dir = './label_img/'
base_dir = './label_img/'

# aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
# aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
# aug = iaa.Add((-40, 40))
# aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)
# aug = iaa.Multiply((0.5, 1.5))
# aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))

# 视频特定颜色追踪
import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa


def augumentor(image):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 建立lambda表达式，
    seq = iaa.Sequential(
        [

            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.GaussianBlur(sigma=(0, 0.5))),

                           # 锐化处理
                           sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),

                           iaa.Affine(rotate=(-1.5, 1.5)),

                           # 加入高斯噪声
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # 每个像素随机加减-10到10之间的数
                           iaa.Add((-10, 10)),

                           # 像素乘上0.5或者1.5之间的数字.
                           iaa.Multiply((0.75, 1.25)),

                           # 将整个图像的对比度变为原来的一半或者二倍
                           iaa.ContrastNormalization((0.6, 1.5)),

                           #                            改变某一通道的值
                           iaa.WithChannels(1, iaa.Add((10, 50))),

                           #                            灰度
                           iaa.Grayscale(alpha=(0.0, 1.0)),

                           #                            加钢印
                           iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),

                           #                            扭曲图像的局部区域
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03)))
                       ],

                       random_order=True  # 随机的顺序把这些操作用在图像上
                       )
        ],
        random_order=True  # 随机的顺序把这些操作用在图像上
    )

    image_aug = seq.augment_image(image)
    return image_aug


# def img_trans(img):
#    seq = iaa.Sequential([
##            iaa.PiecewiseAffine(scale=(0.01, 0.03))
##            iaa.ContrastNormalization((0.6, 1.5))
##            iaa.Multiply((0.75, 1.25))
##            iaa.Affine(rotate=(-1.5, 1.5))
##            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
##            iaa.AdditiveGaussianNoise(
##                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
##                           )
##            iaa.WithChannels(1, iaa.Add((10, 50)))
##            iaa.Grayscale(alpha=(0.0, 1.0))
##            iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
#            iaa.EdgeDetect(alpha=(0.0, 1.0))
#            ])
#    image_aug = seq.augment_image(img)
#    return image_aug


for img in file_name:
    img1 = cv2.imread(base_dir + img)
    imgdir = out_dir + img.replace('.jpg', '')
    print(imgdir)
    image_aug = augumentor(img1)
    #    image_aug = augumentor(img1)
    #    cv2.imwrite(imgdir + '.jpg',image_aug)
    cv2.imwrite(imgdir + str(index) + '.jpg', image_aug)