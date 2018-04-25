import os, sys, json
from os import path

import numpy as np
from PIL import Image
from cntk import load_model
from easydict import EasyDict as edict
    
if __name__ == "__main__":
    import argparse
    import os
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='FRCNN Training')
    
    parser.add_argument('--gpu', type=int, metavar='<integer>', required=False, 
                        help='Specify 1 to use gpu for training.')

    parser.add_argument('--tagged-images', type=str, metavar='<path>',
                        help='Path to image file or to a directory containing tagged image(s) in jpg format', required=True)
    
    parser.add_argument('--num-train', type=int, metavar='<integer>',
                        help='Number of training images. For example: 200',
                        required=True)

    parser.add_argument('--num-epochs', type=int, metavar='<integer>',
                        help='Number of epochs to run training.', required=False)

    args = parser.parse_args()

    from utils.config_helpers import merge_configs
    import utils.od_utils as od

    available_detectors = ['FasterRCNN']

    def download_base_model():
        print("\nDownloading AlexNet base model...")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder,"..", "..", "..", "PretrainedModels"))
        from download_model import download_model_by_name
        download_model_by_name("AlexNet_ImageNet_Caffe")

    download_base_model()

    def create_custom_image_annotations():
        print("\nCreating custom image annotations...")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder, "utils", "annotations"))
        from annotations_helper import create_class_dict, create_map_files
        data_set_path = args.tagged_images
        
        class_dict = create_class_dict(data_set_path)
        create_map_files(data_set_path, class_dict, training_set=True)
        create_map_files(data_set_path, class_dict, training_set=False)

    create_custom_image_annotations()

    def create_custom_config():
        print("\nCreating custom config for your image set in /cntk/Examples/Images/Detection/utils/configs...")
        with open("./utils/configs/custom_image_config.py","w+") as config:
            config.write("""from easydict import EasyDict as edict
__C = edict()
__C.DATA = edict()
cfg = __C
__C.DATA.DATASET = \"CustomImages\"
__C.DATA.MAP_FILE_PATH = \"%s\"
__C.DATA.CLASS_MAP_FILE = \"class_map.txt\"
__C.DATA.TRAIN_MAP_FILE = \"train_img_file.txt\"
__C.DATA.TRAIN_ROI_FILE = \"train_roi_file.txt\"
__C.DATA.TEST_MAP_FILE = \"test_img_file.txt\"
__C.DATA.TEST_ROI_FILE = \"test_roi_file.txt\"
__C.DATA.NUM_TRAIN_IMAGES = %s
__C.DATA.NUM_TEST_IMAGES = 0
__C.DATA.PROPOSAL_LAYER_SCALES = [4, 8, 12]
__C.roi_min_side_rel = 0.04
__C.roi_max_side_rel = 0.4
__C.roi_min_area_rel = 2 * __C.roi_min_side_rel * __C.roi_min_side_rel
__C.roi_max_area_rel = 0.33 * __C.roi_max_side_rel * __C.roi_max_side_rel
__C.roi_max_aspect_ratio = 4.0
""" % (args.tagged_images, args.num_train))

    create_custom_config()

    def get_configuration():
        from utils.config_helpers import merge_configs
        from FasterRCNN_config import cfg as detector_cfg
        from utils.configs.AlexNet_config import cfg as network_cfg
        from utils.configs.custom_image_config import cfg as dataset_cfg
        return merge_configs([detector_cfg, network_cfg, dataset_cfg])

    def run_faster_rcnn():
        print("Running training")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder, "FasterRCNN"))
        from FasterRCNN_train import prepare, train_faster_rcnn
        cfg = get_configuration()
        prepare(cfg, False)
        
        cfg["CNTK"].MAKE_MODE = False
        
        if args.gpu is 1:
            cfg["CNTK"].USE_GPU_NMS = True
        else:
            cfg["CNTK"].USE_GPU_NMS = False

        if not (args.num_epochs is None):
            cfg["CNTK"].E2E_MAX_EPOCHS = args.num_epochs
        
        trained_model = train_faster_rcnn(cfg)
           
    run_faster_rcnn() 
