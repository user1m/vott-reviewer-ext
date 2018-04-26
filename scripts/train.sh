#!/bin/bash

bash -c \
	"source /cntk/activate-cntk && \
	bash -c ' /cntk/Examples/Image/Detection/train.py \
	--tagged-images /cntk/Examples/Image/DataSets/Grocery/grocery/ \
	--num-train 200 \
	--num-epochs 1'"
