#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

if [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
	echo "missing compute option CPU or GPU"
	echo "example usage: ./scripts/build.sh cpu"
	exit 1
fi

if [[ -v "$MODEL_PATH" ]]; then
	echo "MODEL_PATH environment varibale not set"
	echo "first run: export MODEL_PATH=/path/to/cntk/model/"
	exit 1
fi

if [ "$1" == 'gpu' ]; then

	if [ -f '/usr/bin/nvidia-smi' ]; then
		mv /usr/bin/nvidia-cuda-mps-control /usr/bin/nvidia-cuda-mps-control1
		mv /usr/bin/nvidia-cuda-mps-server /usr/bin/nvidia-cuda-mps-server1
		mv /usr/bin/nvidia-debugdump /usr/bin/nvidia-debugdump1
		mv /usr/bin/nvidia-persistenced /usr/bin/nvidia-persistenced1
		mv /usr/bin/nvidia-smi /usr/bin/nvidia-smi1
		mv /usr/bin/nvidia-xconfig /usr/bin/nvidia-xconfig1
	fi

	docker run --rm -it \
		--runtime=nvidia \
		-v $MODEL_PATH:/workdir/model \
		-v $PWD/$WORKDIR/docker/context/flaskapi:/workdir/flaskapi \
		-v $PWD/$WORKDIR/pyfiles:/workdir/pyfiles \
		-v $PWD/$WORKDIR/scripts/:/scripts/ \
		-e PORT=80 \
		-e ENV=development \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-p 3001:80 \
		--name vcntkdev-gpu \
		user1m/vott-reviewer-cntk:$1 \
		bash -c "/scripts/init.sh"

	if [ -f '/usr/bin/nvidia-smi1' ]; then
		mv /usr/bin/nvidia-cuda-mps-control1 /usr/bin/nvidia-cuda-mps-control
		mv /usr/bin/nvidia-cuda-mps-server1 /usr/bin/nvidia-cuda-mps-server
		mv /usr/bin/nvidia-debugdump1 /usr/bin/nvidia-debugdump
		mv /usr/bin/nvidia-persistenced1 /usr/bin/nvidia-persistenced
		mv /usr/bin/nvidia-smi1 /usr/bin/nvidia-smi
		mv /usr/bin/nvidia-xconfig1 /usr/bin/nvidia-xconfig
	fi
else
	docker run --rm -it \
		-v $MODEL_PATH:/workdir/model \
		-v $PWD/$WORKDIR/docker/context/flaskapi:/workdir/flaskapi \
		-v $PWD/$WORKDIR/pyfiles:/workdir/pyfiles \
		-v $PWD/$WORKDIR/scripts/:/scripts/ \
		-e PORT=80 \
		-e ENV=development \
		-p 3001:80 \
		--name vcntkdev-cpu \
		user1m/vott-reviewer-cntk:$1 \
		bash -c "/scripts/init.sh"
fi
