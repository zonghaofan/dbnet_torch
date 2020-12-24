#coding:utf-8
import random
from PIL import Image
import numpy as np
import cv2
import time
import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
from post_seg import SegDetectorRepresenter
import sys, os
import common

class dbnet_code:
    def __init__(self, model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = common.GiB(1)
            # Load the Onnx model and parse it in order to populate the TensorRT network.
            with open(model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
            self.engine = builder.build_cuda_engine(network)
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
    def resize_image(self, img, min_scale=736, max_scale=1088):
        re_im = cv2.resize(img, (min_scale, min_scale))
        return re_im

    def predict(self, img, min_scale=736):
        # with self.engine.create_execution_context() as context:
        img = self.resize_image(img, min_scale=min_scale)
        self.load_normalized_test_case(img, self.inputs[0].host)
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                                             stream=self.stream)
        preds = trt_outputs[0].reshape(1, 2, 736, 736)
        mask = preds[0, 0, ...]
        batch = {'shape': [(736, 736)]}
        box_list, score_list = SegDetectorRepresenter(thresh=0.5, box_thresh=0.7, max_candidates=1000,
                                                      unclip_ratio=1.5)(batch, preds)
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
    def load_normalized_test_case(self, img, pagelocked_buffer):
        # Converts the input image to a CHW Numpy array
        def normalize_image(image):
            # Resize, antialias and transpose the image to CHW.
            # c, h, w = ModelData.INPUT_SHAPE
            # image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS))#.transpose([2, 0, 1])#.astype(trt.nptype(ModelData.DTYPE)).ravel()
            # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
            mean_ = np.array([0.485, 0.456, 0.406])
            std_ = np.array([0.229, 0.224, 0.225])
            img = (image / 255. - mean_) / std_
            return np.asarray(img).transpose([2, 0, 1]).astype(trt.nptype(trt.float32)).ravel()
        # Normalize the image and copy to pagelocked memory.
        np.copyto(pagelocked_buffer, normalize_image(img))
        return img

def draw_bbox(img, result, color=(0, 0, 255), thickness=2):
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img

def debug_main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    onnx_model_file = '/red_detection/DBNet/DBNet_fzh/phone_code_model/model_0.87_depoly.onnx'
    model = dbnet_code(onnx_model_file)
    # path = './第七批手机拍摄错误图片'
    # output_path = './第七批手机拍摄错误图片_条形码检测'

    path = './第四批快递单原始数据手机拍摄'
    output_path = './第四批快递单原始数据手机拍摄_条形码检测'
    # 保存结果到路径
    os.makedirs(output_path, exist_ok=True)
    imgs_list_path = [os.path.join(path, i) for i in os.listdir(path)]
    times = []
    for i, img_list_path, in enumerate(imgs_list_path):
    # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
    # probability that the image corresponds to that label
        print('==img_list_path:', img_list_path)
        img = cv2.imread(img_list_path)
        pred_path = os.path.join(output_path, img_list_path.split('/')[-1].split('.')[0] + '_pred.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st_time = time.time()
        mask, box_list, score_list = model.predict(img)
        times.append(time.time() - st_time)
        cv2.imwrite('./mask.jpg', mask * 255)
        draw_img = draw_bbox(model.resize_image(img, min_scale=736).copy(), box_list)
        cv2.imwrite(pred_path.replace('pred', 'draw'), draw_img)
    print(times)
    print('平均时间为{}'.format(sum(times) / len(times)))

if __name__ == '__main__':
    debug_main()
