# CenterNet

基于Tensorflow实现CenterNet

![image](https://user-images.githubusercontent.com/27406337/191905643-f71c6c24-0ab5-4d15-89ec-ad643b7a51ca.png)


## 项目结构

```
    ├──net：存放centernet的基本网络结构代码，包括resnet、hourglass以及loss
    ├──checkpoint:存放模型训练的checkpoint
    ├──train_dataset：训练数据集
       ├──Annotation：保存目标的标记
       ├──ImageSets：记录训练集、测试集和验证集
       ├──JPEGImages：图像
       ├──train.txt：记录训练集，不同于ImageSets里的train.txt，该文件保存了训练集图像的位置，目标信息
       ├──val.txt：同train.txt
       ├──test.txt：同train.txt
       └──train.names：类别名称
    ├──data：保存基础数据，包括类别名称的文件、字体等
    ├──evaluate：对模型进行评估代码
    ├──logs：训练日志
    ├──models：训练保存的checkpoint所在的文件夹
    ├──utils：一些基础方法：如dataloader、callbacks、fit等
    ├──config.py：配置文件
    ├──inference.py：推理文件
    └──train.py：训练文件
```

## 训练步骤

在`config.py`中设置好训练集路径以及配置好训练的参数之后，执行`train.py`文件即可开始训练。

## 推理验证

1. 在`config.py`配置好你的推理参数，然后运行`inference.py`，查看推理结果



## 20221223更新

1. 增加了pytorch分支，基于pytorch实现CenterNet

## 参考

1. [目标检测 Anchor Free：CenterNet](https://blog.csdn.net/u012655441/article/details/121395058)
