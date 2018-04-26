import os, sys, json
import re
from os import path
import numpy as np
from PIL import Image
from cntk import load_model
from easydict import EasyDict as edict

PAD = 114

def get_classes_description(model_file_path, classes_count):
    model_dir = path.dirname(model_file_path)
    classes_names = {}
    model_desc_file_path = path.join(model_dir, 'class_map.txt')
    if not path.exists(model_desc_file_path):
        # use default parameter names:
        return [ "class_{}".format(i) for i in range(classes_count)]
    with open(model_desc_file_path) as handle:
        class_map = handle.read().strip().split('\n')
        return [class_name.split('\t')[0] for class_name in class_map]

# from https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
numbers = re.compile(r'(\d+)')
def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if __name__ == "__main__":
    import argparse
    import os
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='FRCNN Detector')

    parser.add_argument('--input', type=str, metavar='<path>',
                        help='Path to image file or to a directory containing image in jpg format', required=True)

    parser.add_argument('--output', type=str, metavar='<directory path>',
                        help='Path to output directory', required=False)

    parser.add_argument('--model', type=str, metavar='<file path>',
                        help='Path to model file',
                        required=True)

    parser.add_argument('--cntk-path', type=str, metavar='<dir path>',
                        help='Path to the directory in which CNTK is installed, e.g. c:\\local\\cntk',
                        required=False)

    parser.add_argument('--json-output', type=str, metavar='<file path>',
                        help='Path to output JSON file', required=False)

    args = parser.parse_args()

    print(args)

    if args.cntk_path:
        cntk_path = args.cntk_path
    else:
        cntk_path = "C:\\local\\cntk"
    cntk_scripts_path = path.join(cntk_path, r"Examples/Image/Detection/")
    sys.path.append(cntk_scripts_path)

    from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
    from utils.config_helpers import merge_configs
    import utils.od_utils as od

    available_detectors = ['FasterRCNN']

    def get_configuration(classes):
        # load configs for detector, base network and data set
        from FasterRCNN.FasterRCNN_config import cfg as detector_cfg

        # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
        # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
        from utils.configs.AlexNet_config import cfg as network_cfg
        dataset_cfg = generate_data_cfg(classes)
        return merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': 'FasterRCNN'}])

    def generate_data_cfg(classes):
        cfg = edict({"DATA":edict()})
        cfg.NUM_CHANNELS = 3 # image channels
        cfg["DATA"].CLASSES = classes
        cfg["DATA"].NUM_CLASSES = len(classes)
        return cfg

    def predict(img_path, evaluator, cfg, debug=False):
        # detect objects in single image
        regressed_rois, cls_probs = evaluator.process_image(img_path)
        bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)
        # visualize detections on image
        if debug:
            od.visualize_results(img_path, bboxes, labels, scores, cfg, store_to_path=img_path+"o.jpg")
        # write detection results to output
        fg_boxes = np.where(labels > 0)
        result = []
        for i in fg_boxes[0]:
            # print (cfg["DATA"].CLASSES)
            # print(labels)
            result.append({'label':cfg["DATA"].CLASSES[labels[i]], 'score':'%.3f'%(scores[i]), 'box':[int(v) for v in bboxes[i]]})
        return result

    # from ObjectDetector import predict, get_configuration

    input_path = args.input
    output_path = args.output
    json_output_path = args.json_output
    model_path =  args.model
    model = load_model(model_path)
    FRCNN_DIM_W = model.arguments[0].shape[1]
    FRCNN_DIM_H = model.arguments[0].shape[2]
    labels_count = model.cls_pred.shape[1]
    model_classes = get_classes_description(model_path, labels_count)
    cfg = get_configuration(model_classes)
    evaluator = FasterRCNN_Evaluator(model, cfg)

    if (output_path is None and json_output_path is None):
        parser.error("No directory output path or json output path specified")

    if (output_path is not None) and not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.isdir(input_path):
        import glob
        file_paths = sorted(glob.glob(os.path.join(input_path, '*.jpg')), key=numerical_sort)
    else:
        file_paths = [input_path]

    vott_classes = {model_classes[i]:i for i in range(len(model_classes))}
    if json_output_path is not None:
        json_output_obj = {"classes":vott_classes,
                           "frames" : {}}

    print("Number of images to process: %d"%len(file_paths))

    for file_path, counter in zip(file_paths, range(len(file_paths))):
        width, height = Image.open(file_path).size
        scale = max(width, height)/ max(FRCNN_DIM_W, FRCNN_DIM_H)

        print("Read file in path:", file_path)
        rectangles = predict(file_path, evaluator, cfg)
        regions_list = []
        for rect in rectangles:
            image_base_name = path.basename(file_path)
            x1, y1, x2, y2 = rect["box"]
            if height > width :
                x1, x2 = x1-PAD, x2-PAD
            else:
                y1, y2 = y1-PAD, y2-PAD
            regions_list.append({
                "x1" : int(x1 * scale),
                "y1" : int(y1 * scale),
                "x2" : int(x2 * scale),
                "y2" : int(y2 * scale),
                "class" : vott_classes[rect["label"]]
            })
        json_output_obj["frames"][image_base_name] = {"regions": regions_list}

    if json_output_path is not None:
        print(json_output_path)
        with open(json_output_path, "wt") as handle:
            json_dump = json.dumps(json_output_obj, indent=2)
            handle.write(json_dump)