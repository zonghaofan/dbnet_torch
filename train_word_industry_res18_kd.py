# coding:utf-8
"""
fzh created on 2020/8/26
训练dbnet模型代码
"""
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
import torch
from torch import optim
import argparse
import os
import anyconfig
import torch
import sys
from tqdm.auto import tqdm
import numpy as np
import cv2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import torch.nn as nn
from utils.schedulers import WarmupPolyLR
from utils.util import cal_text_score
from utils.metrics import runningScore
from utils.util import parse_config
from utils.util import Progbar
from data_loader.tools import ICDARCollectFN, get_transforms
from torch.utils.data import DataLoader
from models import build_model, build_loss, build_kd_loss
from data_loader import dataset
from post_processing import get_post_processing


def init_args():
    parser = argparse.ArgumentParser(description='DBNet')
    parser.add_argument('--config_file',
                        default='/red_detection/DBNet/DBNet_fzh/config/icdar2015_resnet18_FPN_DBhead_polyLR_word_industry_kd.yaml',
                        type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')
    args = parser.parse_args()
    return args


def get_f1_score(texts, gt_texts, training_masks, thred=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)

    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)

    y_true = gt_text.flatten()
    y_pred = pred_text.flatten()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1


def get_eval_f1_score(preds, gt_texts):
    pred_text = preds.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy()
    gt_text = gt_text.astype(np.int32)
    y_true = gt_text.flatten()
    y_pred = pred_text.flatten()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1


def get_mask(boxes_batch, img_h, img_w):
    """
    :param boxes_batch: (batch,nums,4,2)
    :param img_h:
    :param img_w:
    :return:
    """
    batch_pred_masks = []
    for pred_boxs in boxes_batch:
        pred_black = np.zeros((img_h, img_w))
        for pred_box in pred_boxs:
            pred_box = np.array(pred_box).astype(np.int32)
            cv2.fillPoly(pred_black, [pred_box], color=1)
        batch_pred_masks.append(pred_black)
    return np.array(batch_pred_masks)


def eval(student_model, optimizer, post_process, validate_loader, output_path):
    """
    :param model:
    :param post_process:
    :param validate_loader:
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    student_model.eval()
    with torch.no_grad():
        P, R, F1 = 0, 0, 0
        for i, batch in enumerate(validate_loader):
            pbar = Progbar(target=len(validate_loader))
            # 数据进行转换和丢到gpu
            batch_size, _, img_h, img_w = batch['img'].shape
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.cuda()
            preds = student_model(batch['img'])
            batch['shape'] = [(img_h, img_w)] * batch_size
            # (batch, nums, 4, 2)
            pred_boxes_batch, scores_batch = post_process(batch, preds, is_output_polygon=False)
            # print('boxes_batch.shape', boxes_batch)
            batch_pred_masks = get_mask(pred_boxes_batch, img_h, img_w)
            precision, recall, f1 = get_eval_f1_score(batch_pred_masks, batch['gt'])
            P += precision
            R += recall
            F1 += f1
            pbar.update(i + 1, values=[('P', P / (i + 1)), ('R', R / (i + 1)), ('F1', F1 / (i + 1))])
        save_model(student_model, optimizer, model_path=os.path.join(output_path,
                                                                     'model_{}.pth'.format(
                                                                         str(round(F1 / (i + 1), 3)))),
                   distributed=False)


def ajust_learning_tri(optimizer, clr_iterations, step_size, base_lr=1e-6, max_lr=1e-4):
    cycle = np.floor(1 + clr_iterations / (2 * step_size))
    x = np.abs(clr_iterations / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


lr_list = []


def train(student_model, teacher_model, optimizer, epochs, student_criterion, teacher_criterion, train_loader, config,
          post_process, validate_loader, output_path):
    for epoch_index in range(epochs):
        student_model.train()
        pbar = Progbar(target=len(train_loader))
        index_train = epoch_index * len(train_loader)
        train_loss = 0.0
        P, R, F1 = 0, 0, 0
        for batch_index, batch in enumerate(train_loader):
            batch_index_ = batch_index
            batch_index_ += index_train
            # lr = optimizer.param_groups[0]['lr']
            lr = ajust_learning_tri(optimizer, batch_index_, step_size=len(train_loader) * 8)
            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.cuda()
            student_preds = student_model(batch['img'])

            loss_dict = {}
            # KD loss
            if teacher_model is not None:
                teacher_model.eval()
                with torch.no_grad():
                    # (b,2,h,w)
                    teacher_outputs = teacher_model(batch['img'])  # shrink_maps, threshold_maps, binary_maps

                kd_loss_dict = teacher_criterion(student_preds, teacher_outputs, batch)
                loss_dict = {**kd_loss_dict}

            gt_loss_dict = student_criterion(student_preds, batch)
            loss_dict = {**gt_loss_dict, **loss_dict}
            # backward
            total_losses = sum(loss_ for loss_ in loss_dict.values())
            optimizer.zero_grad()
            total_losses.backward()
            optimizer.step()

            precision, recall, f1 = get_f1_score(student_preds[:, 0, :, :], batch['shrink_map'], batch['shrink_mask'],
                                                 config['post_processing']['args']['thresh'])
            train_loss += total_losses.item()
            P += precision
            R += recall
            F1 += f1

            pbar.update(batch_index + 1, values=[('loss', train_loss / (batch_index + 1)),
                                                 ('P', P / (batch_index + 1)),
                                                 ('R', R / (batch_index + 1)),
                                                 ('F1', F1 / (batch_index + 1)),
                                                 ('epoch:', epoch_index)])
            lr_list.append(lr)
        if (epoch_index + 1) % 10 == 0:
            eval(student_model, optimizer, post_process, validate_loader, output_path)


def get_trainloader(Dataset, config):
    data_path = config['dataset']['train']['dataset']['args']['data_path'][0]
    img_mode = config['dataset']['train']['dataset']['args']['img_mode']
    pre_processes = config['dataset']['train']['dataset']['args']['pre_processes']
    transform = config['dataset']['train']['dataset']['args']['transforms']
    batch_size = config['dataset']['train']['loader']['batch_size']
    shuffle = config['dataset']['train']['loader']['shuffle']
    pin_memory = config['dataset']['train']['loader']['pin_memory']
    num_workers = config['dataset']['train']['loader']['num_workers']

    train_set = Dataset(data_path=data_path, img_mode=img_mode, pre_processes=pre_processes,
                        filter_keys=['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags'],
                        ignore_tags=['*', '###'], mode='train', transform=get_transforms(transform))
    train_sampler = None
    if config['distributed']:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_set)
        shuffle = False
        pin_memory = True

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, pin_memory=pin_memory,
                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader


def get_evalloader(Dataset, config):
    # print("======config['dataset']['validate']", config['dataset']['validate'])
    data_path = config['dataset']['validate']['dataset']['args']['data_path'][0]
    img_mode = config['dataset']['validate']['dataset']['args']['img_mode']
    min_scale = config['dataset']['validate']['dataset']['args']['pre_processes']['args']['min_scale']
    max_scale = config['dataset']['validate']['dataset']['args']['pre_processes']['args']['max_scale']
    transform = config['dataset']['validate']['dataset']['args']['transforms']
    batch_size = config['dataset']['validate']['loader']['batch_size']
    shuffle = config['dataset']['validate']['loader']['shuffle']
    pin_memory = config['dataset']['validate']['loader']['pin_memory']
    num_workers = config['dataset']['validate']['loader']['num_workers']

    eval_set = Dataset(data_path=data_path, img_mode=img_mode, pre_processes=None,
                       filter_keys=['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags'],
                       ignore_tags=['*', '###'], mode='eval',
                       min_scale=min_scale, max_scale=max_scale, transform=get_transforms(transform))
    eval_sampler = None
    if config['distributed']:
        from torch.utils.data.distributed import DistributedSampler
        eval_sampler = DistributedSampler(eval_set)
        shuffle = False
        pin_memory = True

    eval_loader = DataLoader(dataset=eval_set, sampler=eval_sampler, pin_memory=pin_memory,
                             batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return eval_loader


def weights_add_module(weights):
    from collections import OrderedDict
    modelWeights = OrderedDict()
    for k, v in weights.items():
        name = 'module.' + k  # add `module.`
        modelWeights[name] = v
    return modelWeights


def load_weights(model, optimizer, distributed, checkpoint_path):
    """
    Resume from saved checkpoints
    :param checkpoint_path: Checkpoint path to be resumed
    """
    weights = torch.load(checkpoint_path)
    if distributed:
        module_weights = weights_add_module(weights['state_dict'])
        model.load_state_dict(module_weights)
        logging.info('====载入预训练模型成功====')
    else:
        model.load_state_dict(weights['state_dict'])
        logging.info('====载入预训练模型成功====')
    if optimizer is not None:
        optimizer.load_state_dict(weights['optimizer'])


def save_model(model, optimizer, model_path, distributed=False):
    state_dict = model.module.state_dict() if distributed else model.state_dict()
    # 生成后面要继续训练的模型
    state = {
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, model_path)

    torch.save(state_dict, model_path.replace('.pth', '_depoly.pth'))


def main_entrance():
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    args = init_args()
    config = anyconfig.load(open(args.config_file, 'rb'))
    # print('===config:', config)
    if 'base' in config:
        config = parse_config(config)
    print('===config:', config)
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),
                                             rank=args.local_rank)
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank
    logging.info(config['dataset']['train'])
    student_model = build_model(config['arch']['type'], **config['arch'])
    # print('==student_model:', student_model)
    teacher_model = build_model(config['kd']['type'], **config['kd'])
    print('==teacher_model:', teacher_model)
    #
    student_criterion = build_loss(config['loss'].pop('type'), **config['loss']).cuda()
    teacher_criterion = build_kd_loss(config['kd']['loss'].pop('type'), **config['kd']['loss']).cuda()
    post_process = get_post_processing(config['post_processing'])
    train_loader = get_trainloader(dataset.ICDAR2015Dataset, config)
    eval_loader = get_evalloader(dataset.ICDAR2015Dataset, config)

    student_model = student_model.cuda()
    teacher_model = teacher_model.cuda()
    if config['distributed']:
        student_model = nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank, broadcast_buffers=False,
                                                            find_unused_parameters=True)
        teacher_model = nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank, broadcast_buffers=False,
                                                            find_unused_parameters=True)

    student_checkpoint_path = config['train']['resume_checkpoint']
    teacher_checkpoint_path = config['kd']['resume_checkpoint']
    output_path = config['train']['output_path']
    optimizer = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=0.00005)
    # load_weights(model, optimizer, config['distributed'], checkpoint_path='/red_detection/DBNet/code_pretrain_model/model_latest_express_code_7_13.pth')
    load_weights(student_model, optimizer, config['distributed'], checkpoint_path=student_checkpoint_path)
    load_weights(teacher_model, None, config['distributed'], checkpoint_path=teacher_checkpoint_path)
    epochs = config['train']['epochs']
    warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * len(train_loader)
    # scheduler = WarmupPolyLR(optimizer, max_iters=epochs * len(train_loader),
    #                          warmup_iters=warmup_iters, **config['lr_scheduler']['args'])

    train(student_model, teacher_model, optimizer, epochs, student_criterion, teacher_criterion, train_loader, config,
          post_process, eval_loader, output_path)

    from matplotlib import pyplot as plt

    plt.plot(lr_list)
    plt.savefig('./show_lr_word_industry.png')


def dataloader_debug():
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    # print('===config:', config)
    if 'base' in config:
        config = parse_config(config)
    print('===config:', config)
    print('==torch.cuda.device_count():', torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),
                                             rank=args.local_rank)
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank

    train_loader = get_trainloader(dataset.ICDAR2015Dataset, config)
    # eval_loader = get_evalloader(dataset.ICDAR2015Dataset, config)

    output_path = './查看图片_dataloader'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    epochs = 1
    for epoch in range(epochs):
        for i, data_info in enumerate(tqdm(train_loader)):
            # if i < 1:
            print('===data_info:', data_info.keys())
            batch_img = data_info['img']
            shrink_map = data_info['shrink_map']
            # threshold_label = data_info['threshold_map']
            batch_gt = data_info['gt']
            print('== batch_img.shape', batch_img.shape)
            print('===shrink_map.shape', shrink_map.shape)
            # print(batch_img.shape, threshold_label.shape, threshold_label.shape, batch_gt.shape, data_shape)

            for j in range(batch_img.shape[0]):
                img = batch_img[j].numpy().transpose(1, 2, 0)
                gt = batch_gt[j].numpy() * 255.
                # print('===img.shape:', img.shape)
                # shrink_label = shrink_map[j].numpy()*255.
                gt = np.expand_dims(gt, axis=-1)
                img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.
                img = np.clip(gt + img, 0, 255)
                cv2.imwrite(os.path.join(output_path, str(i) + '_' + str(j) + '.jpg'), img[..., ::-1])
        # break


def debug_model():
    x = torch.rand((8, 3, 640, 640)).cuda()
    model_config = {
        'backbone': {'type': 'resnet18', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model = build_model('Model', **model_config).cuda()
    print(model)
    y = model(x)
    print('y.shape:', y.shape)
    print(model.name)


if __name__ == '__main__':
    main_entrance()
    # dataloader_debug()
    # debug_model()
