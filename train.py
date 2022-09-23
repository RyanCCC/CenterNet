from functools import partial
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.optimizers import Adam
from centernet.centernet import centernet
from utils.callbacks import ExponentDecayScheduler, ModelCheckpoint
from utils.dataloader import CenternetDatasets
from utils.utils import get_classes
import config
import os
from tqdm import tqdm


def get_train_step_fn():
    @tf.function
    def train_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer):
        with tf.GradientTape() as tape:
            loss_value = net([batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices], training=True)
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

@tf.function
def val_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer):
    loss_value = net([batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices])
    return loss_value

def fit_one_epoch(net, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch):
    train_step  = get_train_step_fn()
    total_loss  = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_step:
                break
            batch = [tf.convert_to_tensor(part) for part in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices = batch
            loss_value = train_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer)
            total_loss += loss_value
            pbar.set_postfix(**{'total_loss': float(total_loss) / (iteration + 1), 'lr':optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration>=epoch_step_val:
                break
            # 计算验证集loss
            batch = [tf.convert_to_tensor(part) for part in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices = batch
            loss_value = val_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer)
            val_loss += loss_value
            # 更新验证集loss
            pbar.set_postfix(**{'total_loss': float(val_loss)/ (iteration + 1)})
            pbar.update(1)

    logs = {'loss': total_loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    net.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))

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
        train_dataloader = CenternetDatasets(train_lines, input_shape, batch_size, num_classes, train = True)
        val_dataloader = CenternetDatasets(val_lines, input_shape, batch_size, num_classes, train = False)
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