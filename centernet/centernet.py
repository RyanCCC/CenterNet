import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Model

from centernet.loss import loss
from centernet.hourglass import HourglassNetwork
from centernet.resnet import ResNet50, centernet_head
import colorsys
import os
import time

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.bbox import BBoxUtility
import config

def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(hm, max_objects=100):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 128, 128, 80
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def decode(hm, wh, reg, max_objects=100,num_classes=20):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    reg = tf.reshape(reg, [b, -1, 2])
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
    full_indices = tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) + tf.reshape(indices, [-1])
    topk_reg = tf.gather(tf.reshape(reg, [-1,2]), full_indices)
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])
    
    topk_wh = tf.gather(tf.reshape(wh, [-1,2]), full_indices)
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])

    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]

    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0:1] / 2, topk_cy - topk_wh[..., 1:2] / 2
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0:1] / 2, topk_cy + topk_wh[..., 1:2] / 2
    
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections

def centernet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", num_stacks=2):
    assert backbone in ['resnet50', 'hourglass']
    output_size     = input_shape[0] // 4
    image_input     = Input(shape=input_shape)
    hm_input        = Input(shape=(output_size, output_size, num_classes))
    wh_input        = Input(shape=(max_objects, 2))
    reg_input       = Input(shape=(max_objects, 2))
    reg_mask_input  = Input(shape=(max_objects,))
    index_input     = Input(shape=(max_objects,))

    if backbone=='resnet50':
        C5 = ResNet50(image_input)
        y1, y2, y3 = centernet_head(C5, num_classes)

        if mode=="train":
            loss_ = Lambda(loss, name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
            model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
            return model
        else:
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=detections)
            return prediction_model

    else:
        outs = HourglassNetwork(image_input,num_stacks,num_classes)

        if mode=="train":
            loss_all = []
            for out in outs:  
                y1, y2, y3 = out
                loss_ = Lambda(loss)([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
                loss_all.append(loss_)
            loss_all = Lambda(tf.reduce_mean, name='centernet_loss')(loss_all)

            model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=loss_all)
            return model
        else:
            y1, y2, y3 = outs[-1]
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=[detections])
            return prediction_model


class CenterNet_Inference(object):
    _defaults = {
        "model_path"        : config.model_ckp,
        "classes_path"      : config.class_name,
        "input_shape"       : config.input_shape,
        "backbone"          : config.backbone,
        "confidence"        : config.confidence,
        "nms_iou"           : 0.3,
        #backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
        "nms"               : True,
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(nms_thresh=self.nms_iou)
        self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.centernet = centernet([self.input_shape[0], self.input_shape[1], 3], num_classes=self.num_classes, backbone=self.backbone, mode='predict')
        self.centernet.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

    @tf.function
    def get_pred(self, photo):
        preds = self.centernet(photo, training=False)
        return preds
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.get_pred(image_data).numpy()
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)
        if results[0] is None:
            return image

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]
        font = ImageFont.truetype(font=config.font, size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        # 画图
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box  = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.get_pred(image_data).numpy()
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            outputs = self.get_pred(image_data).numpy()
            results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.get_pred(image_data).numpy()
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)
        if results[0] is None:
            return 

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 