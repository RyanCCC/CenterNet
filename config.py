import os
eager =False
class_name=r'./data/voc.names'
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


# 数据集路径设置
root_path = r'./VOC2007'
train_txt = os.path.join(root_path, 'train.txt')
val_txt = os.path.join(root_path, 'val.txt')

model_ckp = './model/model.h5'
log_dir = './logs/'
font = r'./data/simhei.ttf'
confidence=0.5

