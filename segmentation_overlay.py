import cv2
import os
import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt
import tifffile as tiff
# from libtiff import TIFF
from PIL import Image
import numpy as np

#########################
"""
input: img_seg -> (X,Y)
n : label -- need to be seperated
output: seedMark -> background:0, region 1: 1, region 2 : 2...

TODO: Remove small regions
"""

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img,seed,label,n,seedMark,p = 1):
    height, weight = img.shape
    seedList = []
    seedList.append(seed)

    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            if img[tmpX,tmpY] == n and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    
    return seedMark


def Seperate_Region(img,n):
    seedMark = np.zeros_like(img)
    label = 1
    loc = np.where((img ==n) * (seedMark ==0))
    while(len(loc[0])>0):
        seed = Point(loc[0][0],loc[1][0])
        seedMark = regionGrow(img,seed,label,n,seedMark)
        label = label +1
        loc = np.where((img ==n) * (seedMark ==0))
    # seedMark = remove_small_region(seedMark, min_num= 16)
    seedMark = save_big_region(seedMark)
    # print(seedMark.max(),seedMark.sum())
    return seedMark

def remove_small_region(seedMark,min_num = 1):
    fine_lst = []
    num = 1
    for i in range(seedMark.max()):
        lst = np.where(seedMark == (i+1))
        if len(lst[0])>= min_num:
            seedMark[lst] = num
            num += 1
        else:
            seedMark[lst] = 0
    return seedMark

def save_big_region(seedMark):
	area_lst = []
	for i in range(seedMark.max()):
		lst = np.where(seedMark == (i+1))
		area_lst.append(len(lst[0]))
	max_region = np.argmax(area_lst)
	seedMark[seedMark != (max_region+1)] = 0
	seedMark[seedMark>0] =1
	return seedMark




def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    x = torch.ones(shape)
    result = result.scatter_(1, input, x)

    return result 
def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == (i) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[_mask != 0] += 1
    return _mask

def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return mask
    
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)
        
    return np.array(channels)

def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """
    
    if radius < 0:
        return mask
    
    # We need to pad the borders for boundary conditions
    # print (mask.shape)
    # mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='edge')
    
    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)    
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap
def get_bimask(gt,num_classes):
	edgemapall = []
	for i in range(gt.shape[2]):
		_edgemap = (gt[:,:,i])
		_edgemap = mask_to_onehot(_edgemap, num_classes)
		# print(_edgemap.max())
		_edgemap = onehot_to_binary_edges(_edgemap, 2, num_classes)
		# edgemap = torch.from_numpy(_edgemap).float()
		edgemapall.append(_edgemap)
	edgemapall = np.concatenate(edgemapall,axis = 0)
	# print(edgemapall.shape,"edgemapshape")
	return np.transpose(edgemapall,(1,2,0))


def change_color(img):
	img[:,:,1] = 0
	img = img/2
	return img

def img_dup(img1,img2,percent=1,change = False):
	if change:
		img2 = change_color(img2)
	new_img = img1 + img2*percent
	lst = np.where(new_img>255)
	new_img[lst] = 255
	return new_img.astype(np.uint8)

def my_generate_mask_over_img(path_mask,path_img):
    # img_ori = tiff.imread(path_img)
    # print((img_ori[:,:,1]-img_ori[:,:,0]).sum())
    if 'tif' in path_img:
        img_ori = tiff.imread(path_img)
        img_ori = Image.fromarray(img_ori)
        # img_ori = img_ori.convert('RGB')
        img_ori = np.array(img_ori)
        new_ori_img = np.zeros_like(img_ori)
        new_ori_img[:,:,0] = img_ori[:,:,2]
        new_ori_img[:,:,1] = img_ori[:,:,1]
        new_ori_img[:,:,2] = img_ori[:,:,0]
        img_ori = new_ori_img
    else:
    # cv2.imwrite('./target.png', img)


    	img_ori = cv2.imread(path_img,1)

    # cv2.imshow('a',img_ori*255)
    # cv2.waitKey(0)
    img_mask = cv2.imread(path_mask)
    if img_mask.max() > 0:
    	img_mask = img_mask/img_mask.max()
    img_mask = img_mask.astype(np.int16)
    # print(img_mask.max())

    # new_mask = np.zeros_like(img_mask)
    # temp = Seperate_Region(img_mask[:,:,0],1)
    # new_mask[:,:,1] = temp
    # new_mask[:,:,2] = temp
    # new_mask[:,:,0] = temp
    # img_mask = new_mask
    # cv2.imshow('a',img_mask*255)
    # cv2.waitKey(0)
    # print(img_mask.max())


    # img_ori = img_ori*0
    img_edge = get_bimask(img_mask,1)*255
    img_ori = img_dup(img_ori,img_mask*255,change = True)
    img = img_dup(img_ori,img_edge,1.0)
    return img

# path_seg = './results/PraNet/44old'
# path_img = './data/TestDataset/CVC-300/images'
# path_seg = './data/TestDataset/CVC-300'
# new_path = './video/CVC-300'
# path_seg = './test_ori_result'
# path_img = './test_ori/'
# new_path = './fuse/20230219_1'
# os.makedirs(new_path,exist_ok =True)
# files = os.listdir(path_img)

# for file in files:
#     try:
#     	temp_path_seg = path_seg + '/' + file[:-3]+'png'
#     	temp_path_img = path_img + '/' + file
#     	img = my_generate_mask_over_img(temp_path_seg,temp_path_img)
#     	cv2.imwrite(new_path + '/' + file[:-3]+'jpg', img)
#     	print(file)
#     except Exception as re:
#         print(re)
    # ab