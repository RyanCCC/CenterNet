import numpy as np
from PIL import Image
from pathlib import Path

def check_suffix(file='yolov5s.pth', suffix = ('.pth'), msg = '' ):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            assert Path(f).suffix.lower() in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image   = np.array(image,dtype = np.float32)[:, :, ::-1]
    mean    = [0.40789655, 0.44719303, 0.47026116]
    std     = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model"):
    import os
    from torch.hub import load_state_dict_from_url
    
    if backbone == "hourglass":
        raise ValueError("HourglassNet has no pretrained model")
    
    download_urls = {
        'resnet50'      : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)