import tensorflow as tf
    
def focal_loss(hm_pred, hm_true):
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-6, 1.)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-6, 1.)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    b, c = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
    k = tf.shape(indices)[1]

    y_pred = tf.reshape(y_pred, (b, -1, c))
    length = tf.shape(y_pred)[1]
    indices = tf.cast(indices, tf.int32)
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, k))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) +
                    tf.reshape(indices, [-1]))

    y_pred = tf.gather(tf.reshape(y_pred, [-1,c]),full_indices)
    y_pred = tf.reshape(y_pred, [b, -1, c])

    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss


def loss(args):
    '''
    hm_pred：热力图的预测值       (batch_size, 128, 128, num_classes)
    wh_pred：宽高的预测值         (batch_size, 128, 128, 2)
    reg_pred：中心坐标偏移预测值  (batch_size, 128, 128, 2)
    hm_true：热力图的真实值       (batch_size, 128, 128, num_classes)
    wh_true：宽高的真实值         (batch_size, max_objects, 2)
    reg_true：中心坐标偏移真实值  (batch_size, max_objects, 2)
    reg_mask：真实值的mask        (batch_size, max_objects)
    indices：真实值对应的坐标     (batch_size, max_objects)'''
    hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, indices = args
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    # total_loss = tf.Print(total_loss,[hm_loss,wh_loss,reg_loss])
    return total_loss