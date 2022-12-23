import time

import cv2
import numpy as np
from PIL import Image
import colorsys
import os
import time
from glob import glob
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageDraw, ImageFont

from net.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from tools.tools import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from tools.bbox import decode_bbox, postprocess

class CenterNet(object):
    def __init__(self, **kwargs):
        self._defaults = {
            "model_path" : kwargs['model_path'],
            "classes_path" : kwargs['classes_path'],
            "backbone" : kwargs['backbone'],
            "input_shape" : kwargs['input_shape'],
            "confidence" : kwargs['confidence'],
            "nms_iou" : kwargs['nms_iou'],
            "nms" : kwargs['nms'],
            "letterbox_image" :kwargs['letterbox_image'],
            "cuda" : kwargs['cuda']
        }
        self.__dict__.update(self._defaults)
            
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    def generate(self, onnx=False):
        assert self.backbone in ['resnet50', 'hourglass']
        if self.backbone == "resnet50":
            self.net = CenterNet_Resnet50(num_classes=self.num_classes, pretrained=False)
        else:
            self.net = CenterNet_HourglassNet({'hm': self.num_classes, 'wh': 2, 'reg':2})

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def preprocessing(self, image):
        image = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        return image_data

    def detect(self, image, crop = False, count = False):
        image_shape = np.array(np.shape(image)[0:2])
        image_data = self.preprocessing(image)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]

            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            
            if results[0] is None:
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        font = ImageFont.truetype(font = './data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
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


    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 

        image_shape = np.array(np.shape(image)[0:2])
        image_data = self.preprocessing(image)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            if results[0] is None:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

def video_inference(src, model, save):
    capture = cv2.VideoCapture(src)
    fps = 0.0
    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('./output.mp4', fourcc, 20.0, size)
    while True:
        try:
            t1 = time.time()
            if capture.isOpened():
                ref, frame = capture.read()
                # 获取视频的时间戳
                millseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
                if frame is not None and ref:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(np.uint8(frame))
                    frame = np.array(model.detect(frame))
                    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                    fps  = ( fps + (1./(time.time()-t1)) ) / 2
                    print("fps= %.2f"%(fps))
                    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if save:
                        out.write(frame)
                    cv2.imshow("video",frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
            else:
                break
        except Exception as e:
            print(e)
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()

def dir_inference(imag_dir, model, save=True, save_dir='./result'):
    path_pattern = f'{imag_dir}/*'
    img_number = len(path_pattern)
    result = 0
    for path in glob(path_pattern):
        # 判断是否为文件夹
        if os.path.isdir(path):
            continue
        image = Image.open(path)
        start_time = time.time()
        img = model.detect(image)
        end_time = time.time()
        result += (end_time-start_time)
        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, os.path.basename(path))
            img.save(save_path)
    fps = 1/(result/img_number)
    print(f'finish，fps is {fps}')

if __name__ == "__main__":
    centernet = CenterNet(
        model_path = './model/centernet_resnet50_voc.pth',
        classes_path= './data/voc_classes.names',
        backbone = 'resnet50',
        input_shape = [512, 512],
        confidence = 0.3,
        nms_iou = 0.3,
        nms = True,
        letterbox_image = False,
        cuda = False

    )
    source = './samples/202182193418714.jpg'
    save = False
    save_dir = './result'
    webcam = source.isnumeric() or source.lower().endswith(('.mp4', '.mp3', '.avi')) or source.lower().startswith(('rtsp://', 'rtmp://'))

    if webcam:
        video_inference(source, centernet, False)
    else:
        if os.path.isdir(source):
            dir_inference(source, centernet)
        else:
            image = Image.open(source)
            img = centernet.detect(image)
            img.show()
            if save:
                save_path = os.path.join(save_dir, 'tmp.jpg')
                img.save(save_path)