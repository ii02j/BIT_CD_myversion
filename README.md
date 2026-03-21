# 基于BIT_CD的遥感影像变化检测

## 项目简介

本项目基于BIT_CD模型，在LEVIR-CD数据集上训练，用于遥感影像的变化检测。后续尝试迁移到Sentinel-2影像

## 需求

```
Python 3.6
pytorch 1.6.0
torchvision 0.7.0
einops  0.3.0
```

## 环境配置
-------

## 数据集

- 训练集：LEVIR-CD（建筑变化检测数据集），格式如下图所示：<br>
![image](./images/1.png)

- 数据格式：
```
需改变成如下数据结构：
├─A
├─B
├─label
└─list
```

`A`: t1期图像;

`B`:t2期图像;

`label`: 标注地图;

`list`: 包含train.txt, val.txt and test.txt，每个文件记录变化检测数据集中的图像名称（XXX.png）。

如下图所示：<br>
图二

## 模型结构

模型采用BIT_Transformer,基于ResNet18作为backbone，引入Transformer编码器-解码器结构提取语义token，最终通过特征差分和上采样得到变化图。

**数据流**：
- 输入：两幅256*256 RGB影像
- backbone输出：32通道 64*64 特征图
- 语义token：4个token，每个32维
- Transformer处理后，特征图上采样至256*256
- 分类头输出2通道变化概率图

图五

## 训练

训练命令：
```bash
python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
```

**主要参数：**

- batch_size:8
- max_epochs:200
- lr:0.001

图六

## 训练好的权重

- 验证集精度：
```
-acc: 0.97312
-miou: 0.77472
-mf1: 0.85894
-iou_0（未变化区域）: 0.97209
-iou_1（变化区域）: 0.57734
-F1_0: 0.98585
-F1_1: 0.73204
-precision_0: 0.98503
-precision_1: 0.74374
-recall_0: 0.98667
-recall_1: 0.72070 
```

## 测试与可视化

测试命令：
```bash
python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```

测试结果示例：
图七

## 测试我自己的数据

## 结果截图

图八

## 参考
- https://github.com/justchenhao/BIT_CD

