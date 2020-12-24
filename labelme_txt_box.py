import numpy as np
import cv2
import json
import os
import shutil
def cal_stand_points(points):
    s = np.sum(points, axis=1)
    left_top_index = np.argmin(s)
    right_bottom_index = np.argmax(s)
    rect = np.roll(points, 4-left_top_index, axis=0)
    return rect
def output_json(save_path, im_file, bboxs):
    """
        输入：图片,对应的bbbox左上角顺时针[[x1,y1,x2,y1,x2,y2,x1,y2]]和名字
        输出：labelme json文件
    """
    img = cv2.imread(im_file)
    h, w, _ = img.shape

    im_name = im_file.split('/')[-1]
    # 对应输出json的格式
    jsonaug = {}
    jsonaug['flags'] = {}
    jsonaug['fillColor'] = [255, 0, 0, 128]
    # jsonaug['shapes']
    jsonaug['imagePath'] = im_name

    jsonaug['imageWidth'] = w
    jsonaug['imageHeight'] = h
    shapes = []
    for i, bbox in enumerate(bboxs):
        print('==bbox:', bbox)
        print('type(bbox[0]):', type(bbox[0]))
        temp = {"flags": {},
                "line_color": None,
                # "shape_type": "rectangle",
                "shape_type": "polygon",
                "fill_color": None,
                "label": "red"}
        temp['points'] = []
        temp['points'].append([int(bbox[0]), int(bbox[1])])
        temp['points'].append([int(bbox[2]), int(bbox[3])])
        temp['points'].append([int(bbox[4]), int(bbox[5])])
        temp['points'].append([int(bbox[6]), int(bbox[7])])
        shapes.append(temp)
    print('==shapes:', shapes)
    jsonaug['shapes'] = shapes

    jsonaug['imageData'] = None
    jsonaug['lineColor'] = [0, 255, 0, 128]
    jsonaug['version'] = '3.16.3'

    cv2.imwrite(os.path.join(save_path, im_name), img)

    with open(os.path.join(save_path, im_name.replace('.jpg', '.json')), 'w+') as fp:
        json.dump(jsonaug, fp=fp, ensure_ascii=False, indent=4, separators=(',', ': '))
    return jsonaug



def box_labelme_json():
    path = './第二批手机拍摄条形码标注数据'
    imgs_list_path = [os.path.join(path, i) for i in os.listdir(path) if '.jpg' in i]
    # for i,img_list_path in enumerate(imgs_list_path):
    #     if 'pred'  in img_list_path:
    #         os.remove(img_list_path)
    for i,img_list_path in enumerate(imgs_list_path):
        # if i<1:
            txt_list_path = img_list_path.replace('jpg', 'txt')
            boxs = []
            with open(txt_list_path, 'r', encoding='utf-8') as file:
                for i, read_info in enumerate(file.readlines()):
                    boxs.append(list(map(float,read_info.split(',')[:-1])))
            print(boxs)
            output_json(path, img_list_path, boxs)

#将labelme的box转换成txt box
def lableme_points_txt():
    path = './第二批手机拍摄条形码标注数据'

    output_path = './第二批手机拍摄条形码标注数据_画框'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    imgs_list_path = [os.path.join(path, i) for i in os.listdir(path) if '.jpg' in i]

    for i, img_list_path in enumerate(imgs_list_path):
        # if i<1:
        img = cv2.imread(img_list_path)
        json_list_path = img_list_path.replace('.jpg', '.json')
        output_txt_path = img_list_path.replace('.jpg', '.txt')
        with open(json_list_path, 'r') as file:
            json_info = json.load(file)
        print('===json_info', json_info)
        shapes = json_info['shapes']
        output_points = []
        for shape in shapes:
            points = np.array(shape['points']).astype(np.int)
            cv2.polylines(img, [np.array(points).reshape(-1, 1, 2)], True, (0, 0, 255), thickness=1)
            points = cal_stand_points(points)
            # print('===points', points)
            output_points.append(list(map(str, (points.reshape(-1).tolist()))))
        # print('===output_points', output_points)
        with open(output_txt_path, 'w', encoding='utf-8') as file:
            [file.write(','.join(out) + ""","二维码哈哈哈哈"\n""") for out in output_points]
        cv2.imwrite(os.path.join(output_path, img_list_path.split('/')[-1]), img)

if __name__ == '__main__':
    # box_labelme_json()
    lableme_points_txt()
