import numpy as np
import tensorflow as tf

class BBoxUtility(object):
    def __init__(self, nms_thresh=0.45, top_k=300):
        self._nms_thresh    = nms_thresh
        self._top_k         = top_k

    def bbox_iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        
        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                    np.maximum(inter_rect_y2 - inter_rect_y1, 0)
        
        area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
        
        iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
        return iou

    def centernet_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def postprocess(self, prediction, nms, image_shape, input_shape, letterbox_image, confidence=0.5):
        results = [None for _ in range(len(prediction))]
    
        for i in range(len(prediction)):
            detections              = prediction[i]
            detections[:, [0, 2]]   = detections[:, [0, 2]] / (input_shape[1] / 4)
            detections[:, [1, 3]]   = detections[:, [1, 3]] / (input_shape[0] / 4)
            conf_mask   = detections[:, 4] >= confidence
            detections  = detections[conf_mask]
            
            unique_labels   = np.unique(detections[:, -1])
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                if nms:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = detections_class[:, :4]
                    confs_to_process = detections_class[:, 4]
                    #-----------------------------------------#
                    #   进行iou的非极大抑制
                    #-----------------------------------------#
                    idx = tf.image.non_max_suppression(
                        boxes_to_process, confs_to_process,
                        self._top_k,
                        iou_threshold=self._nms_thresh
                    ).numpy()
                    max_detections  = detections_class[idx]
                    
                else:
                    max_detections  = detections_class
                results[i]      = max_detections if results[i] is None else np.concatenate((results[i], max_detections), axis = 0)

            if results[i] is not None:
                box_xy, box_wh      = (results[0][:, 0:2] + results[0][:, 2:4])/2, results[0][:, 2:4] - results[0][:, 0:2]
                results[0][:, :4]   = self.centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results