#!/bin/bash

bash -c \
	"source /cntk/activate-cntk && \
	bash -c 'python /cntk/Examples/Image/Detection/pyfiles/predict.py \
	--tagged-images /cntk/Examples/Image/DataSets/Grocery/grocery/ \
	--num-test 5 \
	--conf-threshold 0.82'"
