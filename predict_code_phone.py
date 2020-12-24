#coding:utf-8
"""
fzh created on 2020/8/26
推理dbnet模型代码
"""
import os
import sys
import numpy as np
import time
import math
import cv2
import torch

from models import build_model
from post_processing import get_post_processing
from torchvision import transforms

def resize_image(img, min_scale=480, max_scale=480):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(min_scale) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_scale:
        im_scale = float(max_scale) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 32 == 0 else (new_h // 32 + 1) * 32
    new_w = new_w if new_w // 32 == 0 else (new_w // 32 + 1) * 32
    # print('==new_h,new_w:', new_h, new_w)
    re_im = cv2.resize(img, (new_w, new_h))
    return re_im

class Code_model:
    def __init__(self, model_path, post_p_thre=0.7):
        checkpoint = torch.load(model_path)
        model_config = {
            'backbone': {'type': 'resnet18', 'pretrained': False, "in_channels": 3},
            'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
            'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
        }
        self.model = build_model('Model', **model_config).cuda()

        self.post_process = get_post_processing({'type': 'SegDetectorRepresenter',
                                                'args': {'thresh': 0.5, 'box_thresh': 0.7, 'max_candidates': 1000, 'unclip_ratio': 1.7}})
        self.post_process.box_thresh = post_p_thre
        self.img_mode = 'RGB'
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.transform = []
        self.transform = transforms.Compose([
                                            transforms.ToTensor(), # 转为Tensor 归一化至0～1
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                 (0.229, 0.224, 0.225)), # 归一化
                                                                 ])
    def predict(self, img, is_output_polygon=False, min_scale=480, max_scale=480):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, min_scale, max_scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.cuda()
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            preds = self.model(tensor)
            # print('==preds.shape:', preds.shape)#(b,2,h,w)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list

def main_debug():
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_file_list
    model_path = '/red_detection/DBNet/DBNet_fzh/phone_code_model/model_0.88_depoly.pth'
    path = './测试图片'
    output_path = './测试图片_条形码检测'
    debug = True
    # 保存结果到路径
    os.makedirs(output_path, exist_ok=True)

    imgs_list_path = [os.path.join(path, i) for i in os.listdir(path)]
    # 初始化网络
    model = Code_model(model_path)
    TIMes = []
    for i, img_list_path in enumerate(imgs_list_path):
        # if i<1:
        print('==img_list_path:', img_list_path)
        img = cv2.imread(img_list_path)
        st = time.time()
        preds, boxes_list, score_list = model.predict(img, is_output_polygon=False)
        print('每张图片分割耗时:{}'.format(time.time() - st))
        TIMes.append(time.time() - st)
        if debug:
            img = draw_bbox(img, boxes_list)
            pred_path = os.path.join(output_path, img_list_path.split('/')[-1].split('.')[0] + '_pred.jpg')
            cv2.imwrite(os.path.join(output_path, img_list_path.split('/')[-1].split('.')[0]+'.jpg'), img)
            cv2.imwrite(pred_path, preds * 255)
            # save_result(os.path.join(output_path, img_list_path.split('/')[-1]).replace('.jpg', '.txt'), boxes_list,
            #             score_list, False)

if __name__ == '__main__':
    main_debug()