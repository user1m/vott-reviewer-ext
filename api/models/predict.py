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

    parser.add_argument('--gpu', type=int, metavar='<integer>',
                        help='Specify 1 to use gpu in prediction.', required=False)

    parser.add_argument('--tagged-images', type=str, metavar='<path>',
                        help='Path to image file or to a directory containing tagged image(s) in jpg format', required=True)

    parser.add_argument('--model-path', type=str, metavar='<path>',
                        help='Path to pretrained model file', required=False)

    parser.add_argument('--num-test', type=int, metavar='<integer>',
                        help='Number of testing images. For example: 5',
                        required=True)

    parser.add_argument('--conf-threshold', type=float, metavar='<float>',
                        help='Confidence threshold when drawing bounding boxes. Choose a float in this range: [0, 1).', required=False)

    args = parser.parse_args()

    from utils.config_helpers import merge_configs
    import utils.od_utils as od

    available_detectors = ['FasterRCNN']

    json_output = {"images": {}}

    def download_base_model():
        print("\nDownloading AlexNet base model...")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder,"..", "..", "..", "PretrainedModels"))
        from download_model import download_model_by_name
        download_model_by_name("AlexNet_ImageNet_Caffe")

    download_base_model()

    def get_configuration():
        from utils.config_helpers import merge_configs
        from FasterRCNN_config import cfg as detector_cfg
        from utils.configs.AlexNet_config import cfg as network_cfg
        from utils.configs.custom_image_config import cfg as dataset_cfg
        return merge_configs([detector_cfg, network_cfg, dataset_cfg])

    def plot_test_set_results(evaluator, num_images_to_plot, results_base_path, cfg):
        from matplotlib.pyplot import imsave
        from utils.rpn.bbox_transform import regress_rois
        from utils.nms_wrapper import apply_nms_to_single_image_results
        import json

        # get image paths
        with open(cfg["DATA"].TEST_MAP_FILE) as f:
            content = f.readlines()
        img_base_path = os.path.dirname(os.path.abspath(cfg["DATA"].TEST_MAP_FILE))
        img_file_names = [os.path.join(img_base_path, x.split('\t')[1]) for x in content]
        img_shape = (cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)

        print("Plotting results from for %s images." % num_images_to_plot)

        for i in range(0, num_images_to_plot):
            img_path = img_file_names[i]
            out_cls_pred, out_rpn_rois, out_bbox_regr, dims = evaluator.process_image_detailed(img_path)
            labels = out_cls_pred.argmax(axis=1)
            scores = out_cls_pred.max(axis=1)
            # apply regression and nms to bbox coordinates
            regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)
            nmsKeepIndices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                               use_gpu_nms=cfg.USE_GPU_NMS,
							                                   device_id=cfg.GPU_ID,
                                                               nms_threshold=cfg.RESULTS_NMS_THRESHOLD,
                                                               conf_threshold=cfg.RESULTS_NMS_CONF_THRESHOLD)
            filtered_bboxes = regressed_rois[nmsKeepIndices]
            filtered_labels = labels[nmsKeepIndices]
            filtered_scores = scores[nmsKeepIndices]

            json_output["images"][img_path] = { "class": cfg["DATA"].CLASSES[0], "bounding_boxes": []}

            img = visualize_detections(img_path, filtered_bboxes, filtered_labels, filtered_scores,
                                       img_shape[2], img_shape[1],
                                       classes=cfg["DATA"].CLASSES,
                                       draw_negative_rois=cfg.DRAW_NEGATIVE_ROIS,
                                       decision_threshold=cfg.RESULTS_BGR_PLOT_THRESHOLD)
            imsave("{}/{}_regr_{}".format(results_base_path, i, os.path.basename(img_path)), img)

    def ToIntegers(list1D):
        return [int(float(x)) for x in list1D]

    def visualize_detections(img_path, roi_coords, roi_labels, roi_scores,
                             pad_width, pad_height, classes,
                             draw_negative_rois = False, decision_threshold = 0.0):
        from utils.plot_helpers import imWidthHeight, getColorsPalette, drawText
        import cv2
        from PIL import Image, ImageFont, ImageDraw
        import json

        # read and resize image
        imgWidth, imgHeight = imWidthHeight(img_path)
        scale = 800.0 / max(imgWidth, imgHeight)
        imgHeight = int(imgHeight * scale)
        imgWidth = int(imgWidth * scale)
        if imgWidth > imgHeight:
            h_border = 0
            v_border = int((imgWidth - imgHeight)/2)
        else:
            h_border = int((imgHeight - imgWidth)/2)
            v_border = 0

        PAD_COLOR = [103, 116, 123] # [114, 114, 114]
        cv_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_img, (imgWidth, imgHeight), interpolation=cv2.INTER_NEAREST)
        result_img = cv2.copyMakeBorder(resized,v_border,v_border,h_border,h_border,cv2.BORDER_CONSTANT,value=PAD_COLOR)
        rect_scale = 800 / pad_width

        assert(len(roi_labels) == len(roi_coords))
        if roi_scores is not None:
            assert(len(roi_labels) == len(roi_scores))
            minScore = min(roi_scores)
            if minScore > decision_threshold:
                decision_threshold = minScore * 0.5

        # draw multiple times to avoid occlusions
        for iter in range(0,3):
            for roiIndex in range(len(roi_coords)):
                label = roi_labels[roiIndex]
                if roi_scores is not None:
                    score = roi_scores[roiIndex]
                    if decision_threshold and score < decision_threshold:
                        label = 0

                # init drawing parameters
                thickness = 1
                if label == 0:
                    color = (255, 0, 0)
                else:
                    color = getColorsPalette()[label]

                rect = [(rect_scale * i) for i in roi_coords[roiIndex]]
                rect[0] = int(max(0, min(pad_width, rect[0])))
                rect[1] = int(max(0, min(pad_height, rect[1])))
                rect[2] = int(max(0, min(pad_width, rect[2])))
                rect[3] = int(max(0, min(pad_height, rect[3])))

                # draw in higher iterations only the detections
                if iter == 0 and draw_negative_rois:
                    drawRectangles(result_img, [rect], color=color, thickness=thickness)
                elif iter==1 and label > 0:
                    thickness = 4
                    boundaries = drawRectangles(result_img, [rect], color=color, thickness=thickness)
                    for boundary_index in range(len(boundaries)):
                       boundary = boundaries[boundary_index]
                       score = roi_scores[roiIndex]
                       json_output["images"][img_path]["class"] = classes[label]
                       json_output["images"][img_path]["bounding_boxes"].append({"confidence_level": str(score), "bounding_box": boundary})
                    json_output["images"][img_path]["class"] = classes[label]
                elif iter == 2 and label > 0:
                    try:
                        font = ImageFont.truetype(available_font, 18)
                    except:
                        font = ImageFont.load_default()
                    text = classes[label]
                    if roi_scores is not None:
                        text += "(" + str(round(score, 2)) + ")"
                    result_img = drawText(result_img, (rect[0],rect[1]), text, color = (255,255,255), font = font, colorBackground=color)
        return result_img

    def drawRectangles(img, rects, color = (0, 255, 0), thickness = 2):
        import cv2
        boundaries = []
        for rect in rects:
            pt1 = tuple(ToIntegers(rect[0:2]))
            pt2 = tuple(ToIntegers(rect[2:]))
            try:
                cv2.rectangle(img, pt1, pt2, color, thickness)
                boundaries.append({"x1": pt1[0], "x2": pt2[0], "y1": pt1[1], "y2": pt2[1]})
            except:
                print("Unexpected error:", sys.exc_info()[0])
        return boundaries

    def run_faster_rcnn():
        print("Running training")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder, "FasterRCNN"))
        from cntk import load_model
        from FasterRCNN_train import prepare
        from FasterRCNN_eval import compute_test_set_aps
        from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
        import numpy as np
        import json

        cfg = get_configuration()
        prepare(cfg, False)

        cfg["DATA"].NUM_TEST_IMAGES = args.num_test
        cfg["CNTK"].MAKE_MODE = True
        cfg["CNTK"].VISUALIZE_RESULTS = True
        if args.gpu:
            cfg["CNTK"].USE_GPU_NMS = True
        else:
            cfg["CNTK"].USE_GPU_NMS = False
        if not (args.conf_threshold is None):
            cfg.RESULTS_NMS_CONF_THRESHOLD = args.conf_threshold

        trained_model = load_model(cfg["MODEL_PATH"])
        if not (args.model_path is None):
            trained_model = load_model(args.model_path)

        eval_results = compute_test_set_aps(trained_model, cfg)

        for class_name in eval_results: print('Average precision (AP) for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
        print('Mean average precision (AP) = {:.4f}'.format(np.nanmean(list(eval_results.values()))))

        if cfg["CNTK"].VISUALIZE_RESULTS:
            num_eval = min(cfg["DATA"].NUM_TEST_IMAGES, 100)
            results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
            evaluator = FasterRCNN_Evaluator(trained_model, cfg)
            plot_test_set_results(evaluator, num_eval, results_folder, cfg)

        with open(r"/cntk/Examples/Image/Detection/FasterRCNN/Output/custom_images_output.json", "w+") as resultFile:
            print("Bounding boxes written to /cntk/Examples/Image/Detection/FasterRCNN/Output/custom_images_output.json")
            resultFile.write(json.dumps(json_output))

    run_faster_rcnn()
