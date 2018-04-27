from flask import Flask, abort, request
from flask_restful import Resource, Api
from flask import make_response
import os, sys, json, time, uuid, requests, re, io, numpy as np
from os import path
from PIL import Image
from cntk import load_model
from easydict import EasyDict as edict
sys.path.append(
    os.path.join(os.path.dirname("/cntk/Examples/Image/Detection/"), ''))
from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
from utils.config_helpers import merge_configs
import utils.od_utils as od
from flask_cors import CORS, cross_origin

JSON_MIME_TYPE = 'application/json'
HTML_MIME_TYPE = 'text/html'
PAD = 114

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


class ModelLoader:
    def __init__(self):
        self.input_blob = None
        self.image_id = None
        self.model = None
        self.model_classes = None

    def configure(self, model_path=""):
        if self.model is None:
            self.model_path = model_path
            self.model = load_model(self.model_path)
            labels_count = self.model.cls_pred.shape[1]
            self.model_classes = self.get_classes_description(
                self.model_path, labels_count)

    def run(self, blob, id):
        self.input_blob = blob
        self.image_id = id
        # start = time.time()
        cntk_path = "/cntk/"
        #/cntk/Examples/Image/Detection/FasterRCNN/
        cntk_scripts_path = path.join(cntk_path, r"Examples/Image/Detection/")
        sys.path.append(cntk_scripts_path)

        available_detectors = ['FasterRCNN']
        # output_path = self.output
        # model_path = self.model_path
        FRCNN_DIM_W = self.model.arguments[0].shape[1]
        FRCNN_DIM_H = self.model.arguments[0].shape[2]

        cfg = self.get_configuration(self.model_classes)
        evaluator = FasterRCNN_Evaluator(self.model, cfg)
        vott_classes = {
            self.model_classes[i]: i
            for i in range(len(self.model_classes))
        }
        json_output_obj = {"classes": vott_classes, "frames": {}}

        if (len(self.input_blob) > 0):
            img = Image.open(io.BytesIO(self.input_blob))
            file_path = "/workdir/temp/inputs/" + self.image_id + ".jpg"
            img.save(
                file_path, "JPEG", quality=80, optimize=True, progressive=True)
            width, height = img.size
            scale = max(width, height) / max(FRCNN_DIM_W, FRCNN_DIM_H)

            print("Read file in path:", file_path)
            rectangles = self.predict(file_path, evaluator, cfg)
            regions_list = []
            for rect in rectangles:
                image_base_name = path.basename(file_path)
                x1, y1, x2, y2 = rect["box"]
                if height > width:
                    x1, x2 = x1 - PAD, x2 - PAD
                else:
                    y1, y2 = y1 - PAD, y2 - PAD
                regions_list.append({
                    "x1": int(x1 * scale),
                    "y1": int(y1 * scale),
                    "x2": int(x2 * scale),
                    "y2": int(y2 * scale),
                    "class": vott_classes[rect["label"]]
                })
            json_output_obj["frames"][image_base_name] = {
                "regions": regions_list
            }

        json_dump = json.dumps(json_output_obj, indent=2)

        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e, file=sys.stdout)

        return json_dump

    def get_classes_description(self, model_file_path, classes_count):
        model_dir = path.dirname(model_file_path)
        classes_names = {}
        model_desc_file_path = path.join(model_dir, 'class_map.txt')
        if not path.exists(model_desc_file_path):
            # use default parameter names:
            return ["class_{}".format(i) for i in range(classes_count)]
        with open(model_desc_file_path) as handle:
            class_map = handle.read().strip().split('\n')
            return [class_name.split('\t')[0] for class_name in class_map]

    # from https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python

    def numerical_sort(self, value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def get_configuration(self, classes):
        # load configs for detector, base network and data set
        from FasterRCNN.FasterRCNN_config import cfg as detector_cfg
        # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
        # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
        from utils.configs.AlexNet_config import cfg as network_cfg
        dataset_cfg = self.generate_data_cfg(classes)
        return merge_configs([
            detector_cfg, network_cfg, dataset_cfg, {
                'DETECTOR': 'FasterRCNN'
            }
        ])

    def generate_data_cfg(self, classes):
        cfg = edict({"DATA": edict()})
        cfg.NUM_CHANNELS = 3  # image channels
        cfg["DATA"].CLASSES = classes
        cfg["DATA"].NUM_CLASSES = len(classes)
        return cfg

    def blob_predict(self, blob, evaluator, cfg, debug=False):
        # detect objects in single image
        regressed_rois, cls_probs = evaluator.process_image(img_path)
        bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs,
                                                   cfg)
        # visualize detections on image
        if debug:
            od.visualize_results(
                img_path,
                bboxes,
                labels,
                scores,
                cfg,
                store_to_path=img_path + "o.jpg")
        # write detection results to output
        fg_boxes = np.where(labels > 0)
        result = []
        for i in fg_boxes[0]:
            # print (cfg["DATA"].CLASSES)
            # print(labels)
            result.append({
                'label': cfg["DATA"].CLASSES[labels[i]],
                'score': '%.3f' % (scores[i]),
                'box': [int(v) for v in bboxes[i]]
            })
        return result

    def predict(self, img_path, evaluator, cfg, debug=False):
        # detect objects in single image
        regressed_rois, cls_probs = evaluator.process_image(img_path)
        bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs,
                                                   cfg)
        # visualize detections on image
        if debug:
            od.visualize_results(
                img_path,
                bboxes,
                labels,
                scores,
                cfg,
                store_to_path=img_path + "o.jpg")
        # write detection results to output
        fg_boxes = np.where(labels > 0)
        result = []
        for i in fg_boxes[0]:
            # print (cfg["DATA"].CLASSES)
            # print(labels)
            result.append({
                'label': cfg["DATA"].CLASSES[labels[i]],
                'score': '%.3f' % (scores[i]),
                'box': [int(v) for v in bboxes[i]]
            })
        return result


loader = ModelLoader()


class CNTK(Resource):
    @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
    def get(self):
        resp = "<h1> Welcome To VOTT Reviewer Service: <br/> CNTK Endpint</h1>"
        return make_response(resp, 200, {'Content-Type': HTML_MIME_TYPE})

    @app.route('/cntk', methods=['POST'])
    @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
    def post():
        id = str(uuid.uuid4())
        fname = request.values.get("filename") or id
        blob = request.files['image'].read()
        # size = len(blob)
        # print(fname, file=sys.stdout)

        model_path = ""
        for file in os.listdir("/workdir/model/"):
            if file.endswith(".model"):
                model_path = os.path.join("/workdir/model/", file)

        # print(model_path, file=sys.stdout)
        loader.configure(model_path)
        data = loader.run(blob, fname)
        # print(data, file=sys.stdout)
        return make_response(data, 200, {'Content-Type': JSON_MIME_TYPE})


class Home(Resource):
    @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
    def get(self):
        resp = '<html><head></head><body><h1>Welcome To VOTT Reviewer Service</h1> <br/> <p>Apis available:</p><ul><li>get: /</li><li>post: /cntk</li></ul></body></html>'
        return make_response(resp, 200, {'Content-Type': HTML_MIME_TYPE})


api.add_resource(CNTK, '/cntk')
api.add_resource(Home, '/')


@app.errorhandler(404)
def not_found(e):
    return '', 404
