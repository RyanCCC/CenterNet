import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import os
import numpy as np
from inference import CenterNet
import config


def export_onnx(model, file, opset, train, dynamic, simplify, prefix='\033[91m'):
    try:
        import onnx
        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        im = torch.zeros(1, 3, *config.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        torch.onnx.export(
            model, 
            im, 
            file, 
            verbose=False, 
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(file)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                import onnxsim
                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, file)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {file}')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

def export_model(
    weights,
    save_file,
    simplify=False,
    include=('onnx',),
    train=False, # model.train() mode
    dynamic = False, # onnx:dynamic axes
    opset = 12, #ONNX: opset version
    ):
    centernet = CenterNet(
        model_path = weights,
        classes_path= './data/voc_classes.names',
        backbone = 'resnet50',
        input_shape = [512, 512],
        confidence = 0.3,
        nms_iou = 0.3,
        nms = True,
        letterbox_image = False,
        cuda = False
    ).net
    
    include = [x.lower() for x in include]
    if 'onnx' in include:
        print('Exporting onnx model....')
        export_onnx(centernet, save_file, opset, train, dynamic, simplify)

def parse_arg():
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument('--weight', type=str, help='model weight', required=True)
    parser.add_argument('--save_file', type=str, help='save onnx model name', default='./test.onnx')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    return parser

def main(args):
    onnx_save_path = args.save_file
    opset = args.opset
    weight = args.weight
    dynamic = False
    train = False
    simplify = False
    if args.dynamic:
        dynamic = True
    if args.train:
        train = False
    if args.simplify:
        simplify
    export_model(weights=weight, save_file=onnx_save_path, simplify=simplify, train=train, dynamic=dynamic, opset=opset)



if __name__ == '__main__':
    parser = parse_arg()
    args = parser.parse_args()
    main(args=args)