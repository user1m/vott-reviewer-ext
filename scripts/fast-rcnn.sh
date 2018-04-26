#!/bin/bash

# SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")

# if [ -z "$1" ]; then
# 	echo "missing imput image"
# 	exit 1
# fi

# if [ -z "$2" ]; then
# 	echo "missing model path"
# 	exit 1
# fi

source /cntk/activate-cntk &&
	bash -c 'python /cntk/Examples/Image/Detection/fast_rcnn.py \
--input '$1' \
--output /workdir/output/test/ \
--model '$2' \
--cntk-path /cntk/Examples/Image/Detection/FasterRCNN/ \
--json-output '$3''

# source /cntk/activate-cntk &&
# 	bash -c 'python /cntk/Examples/Image/Detection/fast_rcnn.py \
# --input /cntk/Examples/Image/DataSets/Grocery/grocery/positive/WIN_20160803_11_29_07_Pro.jpg \
# --output /workdir/output/test/ \
# --model /workdir/output/faster_rcnn_eval_AlexNet_e2e.model \
# --cntk-path /cntk/Examples/Image/Detection/FasterRCNN/ \
# --json-output /workdir/temp/outputs/json-result.json'
