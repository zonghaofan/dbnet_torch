import numpy as np
import cv2

def shrink_polygon_py(polygon, shrink_ratio):
    """
    对框进行缩放，返回去的比例为1/shrink_ratio 即可
    """
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon

def shrink_polygon_pyclipper(polygon, shrink_ratio):
    from shapely.geometry import Polygon
    import pyclipper
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked

class MakeShrinkMap():
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''

    def __init__(self, min_text_size=8, shrink_ratio=0.4, shrink_type='pyclipper'):
        shrink_func_dict = {'py': shrink_polygon_py, 'pyclipper': shrink_polygon_pyclipper}
        self.shrink_func = shrink_func_dict[shrink_type]
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        shrink_map = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)

        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                if shrinked.size == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                cv2.fillPoly(shrink_map, [shrinked.astype(np.int32)], 1)
                cv2.fillPoly(gt, [polygon.astype(np.int32)], 1)

        data['shrink_map'] = shrink_map
        data['shrink_mask'] = mask
        data['gt'] = gt
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        return cv2.contourArea(polygon)
        # edge = 0
        # for i in range(polygon.shape[0]):
        #     next_index = (i + 1) % polygon.shape[0]
        #     edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])
        #
        # return edge / 2.

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
    # from shapely.geometry import Polygon
    # import pyclipper
    #
    # polygon = np.array([[0, 0], [100, 10], [100, 100], [10, 90]])
    # a = shrink_polygon_py(polygon, 0.4)
    # print(a)
    # print(shrink_polygon_py(a, 1 / 0.4))
    # b = shrink_polygon_pyclipper(polygon, 0.4)
    # print(b)
    # poly = Polygon(b)
    # distance = poly.area * 1.5 / poly.length
    # offset = pyclipper.PyclipperOffset()
    # offset.AddPath(b, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    # expanded = np.array(offset.Execute(distance))
    # bounding_box = cv2.minAreaRect(expanded)
    # points = cv2.boxPoints(bounding_box)
    # print(points)
