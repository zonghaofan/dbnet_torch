import torch
import torch.onnx
import numpy as np
from models import build_model
from post_seg import SegDetectorRepresenter
import cv2
import os
import onnxruntime as ort
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def pth_params_2_ONNX():
    batch_size = 1
    model_config = {
        'backbone': {'type': 'resnet18', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model = build_model('Model', **model_config).cuda()
    model_path = "/red_detection/DBNet/DBNet_fzh/phone_code_model/model_0.87_depoly.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_shape = (3, 736, 736)  # 输入数据,改成自己的输入shape #renet
    example = torch.randn(batch_size, *input_shape, dtype=torch.float32)  # 生成张量
    example = example.cuda()
    export_onnx_file = "/red_detection/DBNet/DBNet_fzh/phone_code_model/model_0.87_depoly.onnx"  # 目的ONNX文件名
    # torch.onnx.export(model, example, export_onnx_file, opset_version = 11, input_names = ["input"], output_names=['output'], verbose=True)
    # torch.onnx.export(model, example, export_onnx_file,\
    #                   opset_version = 10,\
    #                   do_constant_folding = True,  # 是否执行常量折叠优化\
    #                   input_names = ["input"],  # 输入名\
    #                   output_names = ["output"],  # 输出名\
    #                   dynamic_axes = {"input": {0: "batch_size"},# 批处理变量\
    #                     "output": {0: "batch_size"}})
    _ = torch.onnx.export(model,  # model being run
                          example,  # model input (or a tuple for multiple inputs)
                          export_onnx_file,
                          opset_version=10,
                          verbose=False,  # store the trained parameter weights inside the model file
                          training=False,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output']
                          )
def resize_image(img, min_scale=736, max_scale=1088):
    # img_size = img.shape
    # im_size_min = np.min(img_size[0:2])
    # im_size_max = np.max(img_size[0:2])
    #
    # im_scale = float(min_scale) / float(im_size_min)
    # if np.round(im_scale * im_size_max) > max_scale:
    #     im_scale = float(max_scale) / float(im_size_max)
    # new_h = int(img_size[0] * im_scale)
    # new_w = int(img_size[1] * im_scale)
    #
    # new_h = new_h if new_h // 32 == 0 else (new_h // 32 + 1) * 32
    # new_w = new_w if new_w // 32 == 0 else (new_w // 32 + 1) * 32
    # # print('==new_h,new_w:', new_h, new_w)
    re_im = cv2.resize(img, (min_scale, min_scale))
    return re_im
def predict(ort_session, img):
    img = resize_image(img, min_scale=736)
    mean_ = np.array([0.485, 0.456, 0.406])
    std_ = np.array([0.229, 0.224, 0.225])
    img = (img/255. - mean_)/std_
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
    print('==img.shape:', img.shape)
    # st_time = time.time()
    outputs = ort_session.run(None, {'input': img})
    # print('直接run时间', time.time() - st_time)
    b, c, h, w = outputs[0].shape
    mask = outputs[0][0, 0, ...]
    batch = {'shape': [(h, w)]}
    box_list, score_list = SegDetectorRepresenter(thresh=0.5, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5)(batch, outputs[0])
    box_list, score_list = box_list[0], score_list[0]
    is_output_polygon = False
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

    return mask, box_list, score_list
def load_onnx():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = "/red_detection/DBNet/DBNet_fzh/phone_code_model/model_0.87_depoly.onnx"

    ort_session = ort.InferenceSession('/red_detection/DBNet/DBNet_fzh/phone_code_model/model_0.87_depoly.onnx')

    path = './第七批手机拍摄错误图片'
    output_path = './第七批手机拍摄错误图片_条形码检测'
    # 保存结果到路径
    os.makedirs(output_path, exist_ok=True)
    imgs_list_path = [os.path.join(path, i) for i in os.listdir(path)]
    times = []
    nums = 1
    for i in range(nums):
        for i, img_list_path in enumerate(imgs_list_path):
            # if i<1:
                img = cv2.imread(img_list_path)
                pred_path = os.path.join(output_path, img_list_path.split('/')[-1].split('.')[0] + '_pred.jpg')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st_time = time.time()
                mask, boxs, scores = predict(ort_session, img)
                times.append(time.time() - st_time)
                # print('==mask.shape:', mask.shape)
                cv2.imwrite(pred_path, mask * 255)
                draw_img = draw_bbox(resize_image(img, min_scale=736).copy(), boxs)
                cv2.imwrite(pred_path.replace('pred', 'draw'), draw_img)
    print(times)
    print('平均时间为{}'.format(sum(times)/len(times)))
def draw_bbox(img, result, color=(0, 0, 255), thickness=2):
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img

if __name__== "__main__":
    pth_params_2_ONNX()
    load_onnx()