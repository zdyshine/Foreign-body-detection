import os
import numpy as np
import xml.dom.minidom
import shutil

import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold

QXSB = ['bj_bpmh', 'bj_bpps', 'bj_wkps', 'jyz_lw', 'jyz_pl', 'sly_bjbmyw', 'sly_dmyw', 'jsxs',
        'hxq_gjtps', 'hxq_yfps', 'xmbhyc', 'yw_gkxfw', 'yw_nc', 'mcqdmsh', 'gbps', 'gbqs', 'gjptwss',
        'bmwh', 'dthtps', 'yxdghsg']


AQFX = ['yxcr', 'wcaqm', 'wcgz', 'wpdaqs', 'xy', 'rydd', 'hzyw', 'sndmjs', 'qmls','wdls', 'xdwcr']


ZTSB = ['bjdsyc', 'ywzt_yfyc', 'ywzt_ywzsjyc', 'hxq_gjbs', 'kgg_ybh', 'kgg_ybf']

ZCBQ = ['hxq_gjzc', 'jyz_zc', 'xmbhzc', 'aqmzc', 'gzzc', 'aqszc']

Target_Class = ['bj_bpmh', 'bj_bpps', 'bj_wkps', 'bj_zc', 'jyz_lw', 'jyz_pl', 'jyz_zc', 'sly_bjbmyw', 'sly_dmyw',
         'hxq_gjtps', 'hxq_yfps', 'hxq_gjbs', 'hxq_gjzc', 'xmbhyc', 'xmbhzc', 'gbps', 'gbqs','kgg_ybh', 'kgg_ybf']
All_class = Target_Class
# All_class = QXSB + AQFX + ZTSB + ZCBQ
# print(All_class)
# print('all_class:', len(All_class))
# print('QXSB:', len(QXSB))
# print('AQFX:', len(AQFX))
# print('ZTSB:', len(ZTSB))
# print('ZCBQ:', len(ZCBQ))


# if os.path.exists('../all_class/ZCBQ'):
#     print('Exists all_class/ZCBQ')
# else:
#     os.makedirs('../all_class/ZCBQ')
# Base_path = '../all_class/ZCBQ/'
#
# for path in ZCBQ:
#     if os.path.exists(Base_path + path):
#         print(Base_path + path)
#     else:
#         os.makedirs(Base_path + path)

# if os.path.exists('../all_class/new_class'):
#     print('Exists all_class/new_class')
# else:
#     os.makedirs('../all_class/new_class')


def Sort(Path, In_Path, Out_Path):
    '''
    :param Path: xml文件夹的路径
    :param In_Path: xml中对应的图片路径
    :param Out_Path: 相关类别移动到目标文件夹
    '''
    Xml_class = [] # 统计xml文件中的属于总类别的标记数量
    Xml_class_derepeat = [] # 去重复看有多少类
    New_class = [] # 记录未出现过的类别
    New_class_path = [] # 记录未出现过的类别图片路径
    New_xml_path = []  # 记录未出现过的类别xml路径
    Non_Pic = [] # 记录未出现类别的路径

    file_list = os.listdir(Path)
    num = 0 # 记录一共有多少正常标记的图片
    for file in file_list:
        if file.endswith('xml'):
            xmlfile = Path + file
            # print(xmlfile)
            DOMTree = xml.dom.minidom.parse(xmlfile)
            Annotation = DOMTree.documentElement
            filename = Annotation.getElementsByTagName('filename')
            # ImageName = filename[0].childNodes[0].data  # 如果xml文件中的标记文件名能对应
            ImageName = file.replace('.xml', '.jpg')# 如果xml文件中的标记文件名能对应
            objectlist = Annotation.getElementsByTagName('object')
            for objects in objectlist:
                namelist = objects.getElementsByTagName('name')
                name = namelist[0].childNodes[0].data
                if name in All_class:
                    Xml_class.append(name)
                    if name not in Xml_class_derepeat:
                        Xml_class_derepeat.append(name)

                    out_dir = Out_Path + name
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    out_dir = out_dir + '/' + ImageName
                    # print('out_dir:', out_dir)
                    img_dir = In_Path + '/' + ImageName
                    if not os.path.exists(img_dir):
                        img_dir = img_dir.replace('.jpg','.JPG')
                    # print('img_dir:', img_dir)
                    xml_dir_in = Path + file
                    # print('xml_dir_in:', xml_dir_in)
                    xml_dir_out = Out_Path + name + '/' + file
                    # print('xml_dir_out:', xml_dir_out)
                    if os.path.exists(img_dir):
                        shutil.copy(img_dir, out_dir)
                        shutil.copy(xml_dir_in, xml_dir_out)
                    else:
                        Non_Pic.append(img_dir) # 记录路径不存在的原始图片

                    # if name in QXSB:  # 复制图片到QXSB下面的路径中
                    #     out_dir = Out_Path + '/QXSB/' + name + '/' + ImageName
                    #     img_dir = In_Path + '/' + ImageName
                    #     xml_dir_in = Path + file
                    #     xml_dir_out = Out_Path + '/QXSB/' + name + '/' + file
                    #     if os.path.exists(img_dir):
                    #         shutil.copy(img_dir, out_dir)
                    #         shutil.copy(xml_dir_in, xml_dir_out)
                    #     else:
                    #         Non_Pic.append(img_dir) # 记录路径不存在的原始图片
                #     elif name in AQFX: # 复制图片到AQFX下面的路径中
                #         out_dir = Out_Path + '/AQFX/' + name + '/' + ImageName
                #         img_dir = In_Path + '/' + ImageName
                #         xml_dir_in = Path + file
                #         xml_dir_out = Out_Path + '/AQFX/' + name + '/' + file
                #         if os.path.exists(img_dir):
                #             shutil.copy(img_dir, out_dir)
                #             shutil.copy(xml_dir_in, xml_dir_out)
                #         else:
                #             Non_Pic.append(img_dir)
                #     elif name in ZTSB: # 复制图片到AQFX下面的路径中
                #         out_dir = Out_Path + '/ZTSB/' + name + '/' + ImageName
                #         img_dir = In_Path + '/' + ImageName
                #         xml_dir_in = Path + file
                #         xml_dir_out = Out_Path + '/ZTSB/' + name + '/' + file
                #         if os.path.exists(img_dir):
                #             shutil.copy(img_dir, out_dir)
                #             shutil.copy(xml_dir_in, xml_dir_out)
                #         else:
                #             Non_Pic.append(img_dir)
                #     elif name in ZCBQ: # 复制图片到AQFX下面的路径中
                #         out_dir = Out_Path + '/ZCBQ/' + name + '/' + ImageName
                #         img_dir = In_Path + '/' + ImageName
                #         xml_dir_in = Path + file
                #         xml_dir_out = Out_Path + '/ZCBQ/' + name + '/' + file
                #         if os.path.exists(img_dir):
                #             shutil.copy(img_dir, out_dir)
                #             shutil.copy(xml_dir_in, xml_dir_out)
                #         else:
                #             Non_Pic.append(img_dir)

                # else:  # 判断是否有不在给定的类别-----新的类别
                #     num += 1
                #     if name not in New_class: # 记录未出现过的类别
                #         New_class.append(name)
                # #
                # #     # 创建未出现的类别的文件夹
                #     if not os.path.exists(Out_Path + '/new_class/' + name):
                #         # print(Out_Path + '/new_class/' + name)
                #         os.makedirs(Out_Path + '/new_class/' + name)
                #
                #     out_dir = Out_Path + '/new_class/' + name + '/' + ImageName
                #     img_dir = In_Path + '/' + ImageName
                #     if not os.path.exists(img_dir):
                #         img_dir = img_dir.replace('.jpg', '.JPG')
                #     # print('img_dir:', img_dir)
                #     xml_dir_in = Path + file
                #     xml_dir_out = Out_Path + '/new_class/' + name + '/' + file
                #
                #     New_class_path.append(img_dir)
                #     New_xml_path.append(xml_dir_in)
                #
                #     if os.path.exists(img_dir):
                #         shutil.copy(img_dir, out_dir)
                #         shutil.copy(xml_dir_in, xml_dir_out)
                #     else:
                #         Non_Pic.append(img_dir)
    print(num)
    print('Target_CLASS:', len(Target_Class))
    print('Xml_class_derepeat:', len(Xml_class_derepeat))  # 显示正常标记的图片数量
    print('Xml_class_derepeat:', Xml_class_derepeat)
    for cs in Xml_class_derepeat:
        if cs not in Target_Class:
            print(cs)
    print('New_class:', len(New_class)) # 显示未标记的图片数量
    print('Xml_class:', len(Xml_class)) # 显示正常标记的图片数量

    print('len: file_list:', len(file_list)) # xml标记的图片数量
    print('len: all_New_class:', len(New_xml_path))

    print('The num of original file is {}, The class of xml is {}, The new of xml is {}'.format(len(file_list), len(Xml_class_derepeat), len(New_class_path)))
base_dir = '/media/jinhui/1547059EF1791019/户外异物检测/Desktop/'
# Sort(Path=base_dir + '25类缺陷/缺陷08.12/湖南缺陷08.12_xml/',
#          In_Path=base_dir + '25类缺陷/缺陷08.12/湖南缺陷08.12', Out_Path = base_dir + 'all_class/')

# Sort(Path=base_dir + '25类缺陷/样本/guowangyangbenzhengli/14_conv/',
#          In_Path=base_dir + '25类缺陷/样本/guowangyangbenzhengli/14_conv/', Out_Path=base_dir + 'all_class/')

# Sort(Path=base_dir + '25类缺陷/数据样本0822/test_2_xml/',
#          In_Path=base_dir + '25类缺陷/数据样本0822/test_2/', Out_Path=base_dir + 'all_class/test/')

# Sort(Path='/media/jinhui/1547059EF1791019/户外异物检测/Desktop/all_class/new_class/bj/',
#          In_Path='../', Out_Path='../')

def Get_length(Path):
    Name = []
    Length = []
    filelist = os.listdir('../class_data/')
    for file in filelist:
        Name.append(file)
        item = Path + file
        length = len(os.listdir(item))
        # length = file + ':' + str(length)
        Length.append(length)

    write_csv = pd.DataFrame(columns=['Name','Length'])
    write_csv['Name'] = Name
    write_csv['Length'] = Length
    write_csv.to_csv('./TrainData.csv', header=True, index=None, encoding='utf_8_sig')
    print(Length)

# Get_length('../class_data/')

def Get_Label(Path):
    '''
    :param Path: 存放裁剪后的不同类别的文件夹
    :return: 划分后的训练集和验证集
    '''
    # root_dir = '/media/jinhui/1547059EF1791019/户外异物检测/Desktop/'
    if os.path.exists(Path + 'label.txt'):
        os.remove(Path + 'label.txt')
    if os.path.exists(Path + 'train1.txt'):
        os.remove(Path + 'train1.txt')
    if os.path.exists(Path + '/val1.txt'):
        os.remove(Path + '/val1.txt')
    if os.path.exists(Path + '/class_index.txt'):
        os.remove(Path + '/class_index.txt')
    if os.path.exists(Path + '/test1.txt'):
        os.remove(Path + '/test1.txt')
    Label = []
    X = []
    Y = []
    filelist = os.listdir(Path)
    i = 0
    class_index = [] # 存放类别索引
    for file in filelist:
        ImageFile = Path + file
        # print(ImageFile)
        ImagePath = os.listdir(ImageFile)
        for item in ImagePath:
            item = ImageFile + '/' + item + ',' + str(i)
            # print(item)
            Label.append(item)
            X.append(item)
            Y.append(i)
            # print(item)
        index = file + ',' + str(i)
        class_index.append(index)
        i += 1
    # print(Label)
    print(class_index)
    with open(Path + 'label.txt', 'w') as f:
        for item in Label:
            f.write(item + '\n')
    with open(Path + 'class_index.txt', 'w') as ff:
        for item in class_index:
            ff.write(item + '\n')

    # 设置交叉验证
    folds = KFold(n_splits = 10, shuffle=True, random_state=2019) # 0.1的验证
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, X)):
        train_data = list(np.array(X)[trn_idx])
        val_data = list(np.array(X)[val_idx])
    print(len(train_data), len(val_data), len(Label))

    with open(Path + 'train1.txt', 'w') as f1:
        for item in train_data:
            f1.write(item + '\n')

    with open(Path + 'train_val1.txt', 'w') as f2:
        for item in val_data:
            f2.write(item + '\n')

# Get_Label('/media/jinhui/1547059EF1791019/户外异物检测/Desktop/class_crop/' + 'QXSB/') # 传入完整训练路径
Get_Label('/media/jinhui/1547059EF1791019/户外异物检测/Desktop/all_class_crop/') # 传入完整训练路径
# Get_Label('/media/jinhui/1547059EF1791019/户外异物检测/Desktop/test/all_class_crop_test/test/') # 传入完整训练路径

def Val2Test(Path):
    X = []
    Y = []
    path = Path + 'train_val1.txt'
    f = open(path)
    lines = f.readlines()
    for line in lines:
        # print(line.strip())
        x= line.strip()
        X.append(x)
        y = line.strip().split(',')[0]
        Y.append(y)

    folds = KFold(n_splits = 5, shuffle=True, random_state=2019)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, Y)):
        val_data = list(np.array(X)[trn_idx])
        test_data = list(np.array(Y)[val_idx])
    print(len(val_data), len(test_data))

    with open(Path + 'val1.txt', 'w') as f2:
        for item in val_data:
            f2.write(item + '\n')

    with open(Path + 'test1.txt', 'w') as f2:
        for item in test_data:
            f2.write(item + '\n')

# Val2Test('/media/jinhui/1547059EF1791019/户外异物检测/Desktop/class_crop/' + 'QXSB/') # 把验证集划分为验证集和测试集