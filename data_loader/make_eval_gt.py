#coding:utf-8
import numpy as np
import cv2

class MakeGtMap():
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    def __init__(self, min_text_size=8):
        self.min_text_size = min_text_size

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框制作mask
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        gt = np.zeros((h, w), dtype=np.float32)

        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                ignore_tags[i] = True
            else:
                cv2.fillPoly(gt, [polygon.astype(np.int32)], 1)
        data['gt'] = gt
        return data

#得到左上右下
def order_points_clockwise(points):
    s = np.sum(points, axis=1)
    left_top_index = np.argmin(s)
    right_bottom_index = np.argmax(s)
    rect = np.roll(points, 4-left_top_index, axis=0)
    return rect

def main_debug():
    label_path = '/red_detection/DBNet/data/code_train_gt_v2/1.txt'
    img_path = '/red_detection/DBNet/data/code_train_img_v2/1.jpg'
    boxes = []
    texts = []
    ignores = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            # try:
            box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2)).astype(np.float32)  # 顺时针四个点
            # print('==box:', box)
            if cv2.contourArea(box) > 0:
                boxes.append(box)
                label = params[8]
                texts.append(label)
                # print('===self.ignore_tags:', self.ignore_tags)#['*', '###']
                ignores.append(label in ['*', '###'])
            # except:
            #     print('load label failed on {}'.format(label_path))
    print('np.array(boxes):', np.array(boxes))
    img = cv2.imread(img_path)
    data = {
        'text_polys': np.array(boxes),
        'texts': texts,
        'ignore_tags': ignores,
        'img' : img
    }
    import shutil
    shutil.copy(img_path, './')

    makeshrink = MakeShrinkMap()
    new_data = makeshrink(data)
    shrink_map = new_data['shrink_map']
    shrink_mask = new_data['shrink_mask']

    cv2.imwrite('./shrink_map.jpg', shrink_map*255)
    cv2.imwrite('./shrink_mask.jpg', shrink_mask*255)
if __name__ == '__main__':
    main_debug()
