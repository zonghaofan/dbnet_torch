#### 1.DBnet-pytorch

my chinese blog:https://blog.csdn.net/fanzonghao/article/details/107199538

Dbnet is usually  used to detect  word, in fact barcode can be detected.

This project also provide word detect model.

model:

![](https://github.com/zonghaofan/dbnet_torch/blob/master/model.png)

### 2.train

​		follow icdar15 dataset format, x1,y1,x2,y2,x3,y3,x4,y4,label,(x1,y1) is left top,(x2,y2) is right top.<br>

where config/icdar2015_resnet18_FPN_DBhead_polyLR_code_phone.yaml  you can change learning rate,train_path and so on.

single gpu train: python train_code_phone.py

multi gpus train:sh multi_gpu_train.sh  , nedd notice os.environ['CUDA_VISIBLE_DEVICES']  is match  nproc_per_node.

### 3.torch inference

​	python predict_code_phone.py

### 4.tensorrt inference

First python model_to_onnx.py to get onnx model. Then where onnx_project you can  python dbcode_tensorrt_predict.py. 

notice:change model path

### 5.Knowledge Distillation

python train_word_industry_res50.py　train teacher(res50) model;

python train_word_industry_res18_kd.py train student(res18)model;

### 6.labelme json to txt:

​		--change you own path in labelme_txt_box.py<br>
​		python labelme_txt_box.py

### 7.requirements

pytorch1.5

torchvision0.6

cuda9.0+

tensorrt 7.0

### 8.pretrain model

1.word:https://github.com/zonghaofan/dbnet_torch/tree/master/phone_word_model
2.code:https://github.com/zonghaofan/dbnet_torch/tree/master/phone_code_model

### 9.some examples

  1. learning rate　show

     ![](https://github.com/zonghaofan/dbnet_torch/blob/master/show_lr.png)

     ２.some test examples

     ![](https://github.com/zonghaofan/dbnet_torch/blob/master/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87_%E6%9D%A1%E5%BD%A2%E7%A0%81%E6%A3%80%E6%B5%8B/1000.jpg)

3.train loss

![](https://github.com/zonghaofan/dbnet_torch/blob/master/train_loss.png)



### 10.reference

    1. https://github.com/WenmuZhou/DBNet.pytorch



### 11.to do

More tensortrt inference.
