#coding:utf-8
import pathlib
import os
import sys
import cv2
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from .tools import order_points_clockwise, get_datalist, load, expand_polygon
from .iaa_augment import IaaAugment
from .random_crop_data import EastRandomCropData, PSERandomCrop
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap
from .make_eval_gt import MakeGtMap

# from tools import order_points_clockwise, get_datalist, load, expand_polygon
# from iaa_augment import IaaAugment
# from random_crop_data import EastRandomCropData, PSERandomCrop
# from make_border_map import MakeBorderMap
# from make_shrink_map import MakeShrinkMap
# from make_eval_gt import MakeGtMap
import PIL
import copy
class ICDAR2015Dataset(Dataset):
    def __init__(self, data_path, img_mode, pre_processes, filter_keys, ignore_tags, mode='train', min_scale=736, max_scale=1088, transform=None, target_transform=None,**kwargs):
        assert img_mode in ['RGB', 'BRG', 'GRAY']
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contains {}'.format(item_keys)
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.aug = []
        self.mode = mode
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_transform = target_transform
        self._init_pre_processes(pre_processes)
    def _init_pre_processes(self, pre_processes):
        if pre_processes is not None:
            for pre_process in pre_processes:
                # print('=====pre_process======:', pre_process)
                if 'args' not in pre_process:
                    args = {}
                else:
                    args = pre_process['args']
                # print("===pre_process['type']:",  pre_process['type'])
                if isinstance(args, dict):
                    cls = eval(pre_process['type'])(**args)
                else:
                    cls = eval(pre_process['type'])(args)
                self.aug.append(cls)
            # print('==self.aug:', self.aug)
    def load_data(self, data_path):
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        # print('====len(t_data_list):', len(t_data_list))
        # print('===t_data_list[0]', t_data_list[0])
        return t_data_list

    def _get_annotation(self, label_path):
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))#顺时针四个点
                    # print('==box:', box)
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[8]
                        texts.append(label)
                        # print('===self.ignore_tags:', self.ignore_tags)#['*', '###']
                        ignores.append(label in self.ignore_tags)
                except:
                    print('load label failed on {}'.format(label_path))
        data_info = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores
        }
        return data_info
    def apply_pre_processes(self, data_info):
        for aug in self.aug:
            data_info = aug(data_info)
        return data_info
    def __getitem__(self, index):
        try:
            data_info = copy.deepcopy(self.data_list[index])#注意这里要用深拷贝
            # print('====data_info===:', data_info)
            img = cv2.imread(data_info['img_path'], 1 if self.img_mode != 'GRAY' else 0)
            if self.img_mode == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data_info['img'] = img
            if self.mode == 'train':
                data_info = self.apply_pre_processes(data_info)
            else:
                data_info = self.resize_image(data_info)
                # print("===data_info['gt']", data_info['gt'].shape)
            if self.transform:
                data_info['img'] = self.transform(data_info['img'])
            data_info['text_polys'] = data_info['text_polys'].tolist()
            # print("====data_info['text_polys']:", data_info['text_polys'])
            # print('==data_info.keys()', data_info.keys())
            if len(self.filter_keys):
                data_dict = {}
                for k, v in data_info.items():
                    if k not in self.filter_keys:#['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags', 'shape']
                        data_dict[k] = v
                return data_dict
            else:
                return data_info
        except:
            return self.__getitem__(np.random.randint(self.__len__()))
    def __len__(self):
        return len(self.data_list)

    def resize_image(self, data_info):
        img = data_info['img']
        text_polys = data_info['text_polys']#(nums, 4, 2)

        ori_h, ori_w, _ = img.shape
        im_size_min = min(ori_h, ori_w)
        im_size_max = max(ori_h, ori_w)

        im_scale = float(self.min_scale) / float(im_size_min)
        if np.round(im_scale * im_size_max) > self.max_scale:
            im_scale = float(self.max_scale) / float(im_size_max)
        new_h = int(ori_h * im_scale)
        new_w = int(ori_w * im_scale)

        new_h = new_h if new_h // 32 == 0 else (new_h // 32 + 1) * 32
        new_w = new_w if new_w // 32 == 0 else (new_w // 32 + 1) * 32
        # print('==new_h,new_w:', new_h, new_w)
        re_im = cv2.resize(img, (new_w, new_h))

        w_ratio = new_w / ori_w
        h_ratio = new_h / ori_h
        for i in range(len(text_polys)):
            text_polys[i][:, 0] = text_polys[i][:, 0]*w_ratio
            text_polys[i][:, 1] = text_polys[i][:, 1]*h_ratio

        data_info['img'] = re_im

        return MakeGtMap()(data_info)

if __name__ == '__main__':
    import torch
    import anyconfig
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tools import parse_config, ICDARCollectFN, get_transforms
    data_path = '/red_detection/DBNet/data/code_train_v2.txt'
    img_mode = 'RGB'
    pre_processes = [{'type': 'IaaAugment', 'args': [{'type': 'Fliplr', 'args': {'p': 0.5}},
                                                     {'type': 'GaussianBlur', 'args': {'sigma': [0.0, 3.0]}},
                                                     {'type': 'Affine', 'args': {'rotate': [0, 360], 'scale': [0.7, 1.0]}},
                                                     {'type': 'Resize', 'args': {'size': [0.5, 3]}}]},
                     {'type': 'EastRandomCropData', 'args': {'size': [640, 640], 'max_tries': 50, 'keep_ratio': True}},
                     {'type': 'MakeBorderMap', 'args': {'shrink_ratio': 0.4, 'thresh_min': 0.3, 'thresh_max': 0.7}},
                     {'type': 'MakeShrinkMap', 'args': {'shrink_ratio': 0.4, 'min_text_size': 8}}]
    transform = [{'type': 'ToTensor', 'args': {}},
                 {'type': 'Normalize', 'args': {'mean': [0.485, 0.456, 0.406],
                                                'std': [0.229, 0.224, 0.225]}}]

    # train_data = ICDAR2015Dataset(data_path=data_path, img_mode=img_mode, pre_processes=pre_processes,
    #                               filter_keys=['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags', 'shape'],
    #                               ignore_tags=['*', '###'], mode='train', transform=get_transforms(transform))
    # train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)

    eval_data = ICDAR2015Dataset(data_path=data_path, img_mode=img_mode, pre_processes=pre_processes,
                                  filter_keys=['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags', 'shape'],
                                  ignore_tags=['*', '###'], mode='eval', transform=get_transforms(transform))
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False, num_workers=0)
    output_path = './查看图片'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    epochs = 5
    for epoch in range(epochs):
        # for i, data_info in enumerate(tqdm(train_loader)):
        for i, data_info in enumerate(tqdm(eval_loader)):
            # if i < 1:
            #     print('===data_info:', data_info.keys())
                batch_img = data_info['img']
                # shrink_label = data_info['shrink_map']
                # threshold_label = data_info['threshold_map']
                batch_gt = data_info['gt']
                # print(batch_img.shape, threshold_label.shape, threshold_label.shape, batch_gt.shape)
                for j in range(batch_img.shape[0]):
                    img = batch_img[j].numpy().transpose(1, 2, 0)
                    gt = batch_gt[j].numpy()*255.
                    gt = np.expand_dims(gt, axis=-1)
                    img = (img*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))*255.
                    img = np.clip(gt + img, 0, 255)
                    cv2.imwrite(os.path.join(output_path, str(i) + '_' + str(j)+'.jpg'), img[..., ::-1])
            # break
