#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

if [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
	echo "missing compute option CPU or GPU"
	echo "example usage: ./scripts/build.sh cpu"
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

	docker run --rm -itd \
		--runtime=nvidia \
		-v $PWD/$WORKDIR/Output:/workdir/model \
		-e PORT=80 \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-p 3000:80 \
		--name vcntkprod-gpu \
		user1m/vott-reviewer-cntk:$1

	if [ -f '/usr/bin/nvidia-smi1' ]; then
		mv /usr/bin/nvidia-cuda-mps-control1 /usr/bin/nvidia-cuda-mps-control
		mv /usr/bin/nvidia-cuda-mps-server1 /usr/bin/nvidia-cuda-mps-server
		mv /usr/bin/nvidia-debugdump1 /usr/bin/nvidia-debugdump
		mv /usr/bin/nvidia-persistenced1 /usr/bin/nvidia-persistenced
		mv /usr/bin/nvidia-smi1 /usr/bin/nvidia-smi
		mv /usr/bin/nvidia-xconfig1 /usr/bin/nvidia-xconfig
	fi

else

	docker run --rm -itd \
		-v $PWD/$WORKDIR/Output:/workdir/model \
		-e PORT=80 \
		-p 3000:80 \
		--name vcntkprod-cpu \
		user1m/vott-reviewer-cntk:$1
fi
