## 仓库说明：

本仓库是针对基于EASY-EAI-Nano(RV1126)从PC端模型训练、模型单步测试、pytorch模型转换为onnx模型的流程说明，并以口罩检测为例子说明。而模型如何部署到硬件主板上，完整的在线文档教程可以查看以下在线文档的链接：

## 环境说明：

python version >= 3.6

pytorch version >= 1.7

onnx verison >= 1.11

## 准备数据
口罩检测数据百度链接：https://pan.baidu.com/s/1vtxWurn1Mqu-wJ017eaQrw 提取码：6666 

数据集解压后(脚本在数据集里面)，执行以下脚本生成train.txt和valid.txt：
```python
python list_dataset_file.py
```


## 训练模型
训练一个口罩检测模型，需要修改"data/mask.yaml"里面的train.txt和valid.txt的路径。训练脚本如下所示：
```python
python train.py --data mask.yaml --cfg yolov5s.yaml --weights "" --batch-size 64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```
训练完成后会在

## 模型预测
测试训练好的模型：
```python
python detect.py --source data/images --weights ./runs/train/exp/weights/best.pt --conf 0.5
```
测试结果会在"runs/detect"生成：
<img src="./photo/image.jpg">


## 模型导出
执行以下指令把pt模型转换为onnx模型，同时会生成best.anchors.txt：
```python
python export.py --include onnx --rknpu RV1126 --weights ./runs/train/exp/weights/best.pt
```


### EASY-EAI-Nano基于NPU运行速度测试(单位:ms)：

| 模型(640x640输入)         | EASY-EAI-Nano(RV1126)  |
| :---------------------- | :-----------------------------------------:  |
| yolov5s int8量化 |   52    |
| yolov5m int8量化 |   93    |



## 参考库：

https://github.com/ultralytics/yolov5

https://github.com/soloIife/yolov5_for_rknn


## 技术交流群：

QQ群：810456486



