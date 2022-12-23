import os

# 训练参数
cuda =False
distributed = False
fp16 = False

ckp_path = './checkpoint/centernet_resnet50_voc.pth'
input_shape = [512, 512]
backbone='resnet50'
# 冻结训练
Freeze_train = True
Freeze_Epoch = 100
Freeze_batch_size = 16
learning_rate = 1e-3
UnFreeze_Epoch = 100


# 数据集路径设置
root_path = r'./VOC2007'
train_txt = os.path.join(root_path, 'train.txt')
val_txt = os.path.join(root_path, 'val.txt')
class_name=r'./data/voc.names'

log_dir = './logs/'

