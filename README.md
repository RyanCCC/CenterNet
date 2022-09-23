# CenterNet

基于Tensorflow实现CenterNet

![image](https://user-images.githubusercontent.com/27406337/191905643-f71c6c24-0ab5-4d15-89ec-ad643b7a51ca.png)


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

在```config.py```中设置好训练集路径以及配置好训练的参数之后，即可开始训练。
```python
eager =False
class_name=r'D:\Code\AGCIMAIGit\Dataset_Object_Detection\village\village.names'
ckp_path = './checkpoint/centernet_resnet50_voc.h5'
input_shape = [512, 512]
backbone='resnet50'
Freeze_train = True
# 冻结阶段参数
Freeze_Epoch = 100
Freeze_batch_size = 16
Freeze_lr = 1e-3

# 解冻阶段参数
UnFreeze_Epoch = 100
Unfreeze_batch_size = 8
Unfreeze_lr = 1e-4

# 是否开启多线程
num_worker = 1
```
需要注意的地方：

1. 随着batchsize增大，可以将学习率调大，这样可以快速收敛。像我这种学习率调得很小的轻快，收敛真的非常慢
![image](https://user-images.githubusercontent.com/27406337/142379414-857c380e-df73-4fab-a7ad-53e118f9d91d.png)

2. 训练过程使用```transfer learning```，一开始冻结部分的网络进行训练，后续解冻再进行训练。这个完全看个人的喜好。
3. ```num_workers```参数用于设置加载数据的时候是否使用多线程，其实我觉得作用并不大。


# 推理验证
1. 在`config.py`配置好你的推理参数，然后运行`inference.py`，查看推理结果
2. 在evaluate文件夹里面有cal_map.py文件对模型的性能进行评估。

# 关于CenterNet
可以参考我的博客：https://blog.csdn.net/u012655441/article/details/121395058
