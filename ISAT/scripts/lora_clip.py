# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np
from PIL import Image
from skimage.draw.draw import polygon
import os
import json
from tqdm import tqdm

# 要迭代的目录路径
dir = 'C:/Users/xie/Desktop/cuslabel/6_indomitable/'
txts = []
for filename in os.listdir(dir):
    path_json = os.path.join(dir, filename)
    if os.path.isfile(path_json) and filename.lower().endswith(".json"):
        with open(path_json, "r") as file:
            txts.append(json.load(file))
for txt in tqdm(txts, desc='Processing images', position=1) :  
    path_img = os.path.join(dir,txt["info"]["name"])
    imgname = os.path.splitext(txt["info"]["name"])
    img = np.array(Image.open(path_img))
    rbga = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    rbga[:,:,:3] = img
    rpg_mask = np.zeros((img.shape[0], img.shape[1],1),dtype=np.uint8)
    for i, obj in enumerate(tqdm(txt["objects"], desc=f'Processing {txt["info"]["name"]}', total=len(txt["objects"]), position=0)):
        rpg_mask[polygon([int(it[1]) for it in obj["segmentation"]], [int(it[0]) for it in obj["segmentation"]], img.shape)] = 255
    rbga[:,:, 3] = rpg_mask[:, :, 0]
    # 找到Alpha通道大于0的区域的边界,裁剪数组以保留有效数据
    min_row, max_row = np.where(np.any(rbga[:,:, 3] > 0, axis=1))[0][[0, -1]]
    min_col, max_col = np.where(np.any(rbga[:,:, 3] > 0, axis=0))[0][[0, -1]]
    rgba_crop = rbga[min_row:max_row+1, min_col:max_col+1]
    # 发现stable diffsuion不需要透明度
    imgfile = Image.fromarray(rgba_crop[:,:,:3].astype(np.uint8), mode='RGB')
    save_path = os.path.join(dir,".output",obj["category"],os.path.splitext(txt["info"]["name"])[0]+"_"+str(i)+".png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imgfile.save(save_path)