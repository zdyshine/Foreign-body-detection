import os
import xml.dom.minidom
import numpy as np
from PIL import Image
import cv2

base_path = '/media/jinhui/1547059EF1791019/户外异物检测/Desktop/'

def proccess_bar(percent, start_str = '', end_str = '', total_length = 20):
    # 打印进度条
    bar = ''.join(['='] * int(percent * total_length)) + '>'
    bar = '\r' + start_str + '{:0>4.1f}% ['.format(percent*100) + bar.ljust(total_length) + ']' + end_str
    print(bar, end = '', flush = True)

def Crop(Path, SavePath, file):
    '''
    从整图crop出对应类别的图
    :param Path: 从xml文件中取出对应类别的文件夹
    :param SavePath: 保存crop出类别的主文件，里面是不同的子类文件夹
    :param file: 子类文件夹，里面是图片和xml文件
    :return:
    '''
    if len(os.listdir(Path)) == 0:
        print(file + ' is empty')
    else:
        Path_list = os.listdir(Path)
        i = 0
        length = len(Path_list) // 2
        for path in Path_list:
            if path.endswith('xml'):
                xmlfile = Path + path
                imgfile = Path + path.replace('xml', '') + 'jpg'
                if not os.path.exists(imgfile):
                    imgfile = Path + path.replace('xml', '') + 'JPG'
                DOMTree = xml.dom.minidom.parse(xmlfile)  # 打开xml文档
                annotation = DOMTree.documentElement  # 得到文档元素对新
                objectlist = annotation.getElementsByTagName('object')  # 读取object
                for objects in objectlist:
                    namelist = objects.getElementsByTagName('name')
                    name = namelist[0].childNodes[0].data
                    if name == file: # 判断xml读取到的类别，只读取需要的
                        bndbox = objects.getElementsByTagName('bndbox')  # 获取bndbox
                        for box in bndbox:
                            x1_list = box.getElementsByTagName('xmin')
                            x1 = int(x1_list[0].childNodes[0].data)
                            y1_list = box.getElementsByTagName('ymin')
                            y1 = int(y1_list[0].childNodes[0].data)
                            x2_list = box.getElementsByTagName('xmax')
                            x2 = int(x2_list[0].childNodes[0].data)
                            y2_list = box.getElementsByTagName('ymax')
                            y2 = int(y2_list[0].childNodes[0].data)
                            # print(x1, y1, x2, y2)

                            w = x2 - x1
                            h = y2 - y1
                            obj = np.array([x1, y1, x2, y2])  # 设置偏移
                            shift = np.array([[1, 1, 1, 1]])
                            XYmatrix = np.tile(obj, (1, 1))
                            cropboxes = XYmatrix * shift

                            if not os.path.exists(SavePath):
                                os.makedirs(SavePath)
                            img = Image.open(imgfile)
                            for cropbox in cropboxes:
                                cropedimg = img.crop(cropbox)
                                # cropedimg.show()
                                cropedimg.save(SavePath + path.replace('xml', '') + 'jpg')
                proccess_bar(i / length, start_str='', end_str='', total_length=20)
                i += 1
# 提前创建好文件夹： crop_class/QXSB 或者其它大类
filelist = os.listdir(base_path + 'all_class/')
for file in filelist:
    print(file)
    Crop(Path = base_path + 'all_class/' + file + '/', SavePath = base_path + 'all_class_crop/'+ file + '/', file = file)