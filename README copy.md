# Centernet

基于Tensorflow2实现centerNet

# 项目结构
```
    ├──centernetwork：存放centernetwork的基本网络结构代码，包括resnet、hourglass以及loss
    ├──checkpoint:存放模型训练的checkpoint
    ├──traindataset：训练数据集
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

# 训练步骤

# 推理步骤

# 关于CenterNet

