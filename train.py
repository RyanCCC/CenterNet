from functools import partial
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.optimizers import Adam
from centernetwork.centernet import centernet
from utils.callbacks import ExponentDecayScheduler, ModelCheckpoint
from utils.dataloader import CenternetDatasets
from utils.utils import get_classes
from utils.fit import fit_one_epoch
import config
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
    
if __name__ == "__main__":
    eager = config.eager
    classes_path  = config.class_name
    model_path = config.ckp_path
    input_shape = config.input_shape
    backbone  = config.backbone
    Init_Epoch = 0
    Freeze_Epoch = config.Freeze_Epoch
    Freeze_batch_size = config.Freeze_batch_size
    Freeze_lr = config.Freeze_lr
    UnFreeze_Epoch = config.UnFreeze_Epoch
    Unfreeze_batch_size = config.Unfreeze_batch_size
    Unfreeze_lr = config.Unfreeze_lr
    Freeze_Train = config.Freeze_train
    train_annotation_path   = config.train_txt
    val_annotation_path = config.val_txt
    class_names, num_classes = get_classes(classes_path)

    model = centernet([input_shape[0], input_shape[1], 3], num_classes=num_classes, backbone=backbone, mode='train')
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    logging = TensorBoard(log_dir = config.log_dir)
    checkpoint = ModelCheckpoint(config.model_ckp,monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    reduce_lr = ExponentDecayScheduler(decay_rate = 0.94, verbose = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 读取数据集
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if backbone == "resnet50":
        freeze_layer = 171
    elif backbone == "hourglass":
        freeze_layer = 624
    else:
        raise ValueError('Unsupported backbone - `{}`, Use resnet50, hourglass.'.format(backbone))
    if Freeze_Train:
        for i in range(freeze_layer): model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layer, len(model.layers)))
    if True:
        batch_size  = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch  = Freeze_Epoch
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        train_dataloader    = CenternetDatasets(train_lines, input_shape, batch_size, num_classes, train = True)
        val_dataloader      = CenternetDatasets(val_lines, input_shape, batch_size, num_classes, train = False)
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if eager:
            gen_train = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

            gen_train  = gen_train.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model, optimizer, epoch, epoch_step, epoch_step_val, gen_train, gen_val, end_epoch)
        else:
            model.compile(optimizer = Adam(lr), loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
            model.fit_generator(
                generator= train_dataloader,
                steps_per_epoch = epoch_step,
                validation_data = val_dataloader,
                validation_steps = epoch_step_val,
                epochs = end_epoch,
                initial_epoch = start_epoch,
                callbacks = [logging, checkpoint, reduce_lr, early_stopping]
            )
    if Freeze_Train:
        for i in range(freeze_layer): model.layers[i].trainable = True

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataloader = CenternetDatasets(train_lines, input_shape, batch_size, num_classes, train = True)
        val_dataloader = CenternetDatasets(val_lines, input_shape, batch_size, num_classes, train = False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if eager:
            gen_train = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

            gen = gen_train.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch(model, optimizer, epoch, epoch_step, epoch_step_val, gen_train, gen_val, end_epoch)
        else:
            model.compile(optimizer = Adam(lr), loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
            model.fit_generator(
                generator = train_dataloader,
                steps_per_epoch = epoch_step,
                validation_data = val_dataloader,
                validation_steps = epoch_step_val,
                epochs = end_epoch,
                initial_epoch = start_epoch,
                callbacks = [logging, checkpoint, reduce_lr, early_stopping]
            )