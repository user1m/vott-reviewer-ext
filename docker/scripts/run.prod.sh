#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

if [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
	echo "missing compute option CPU or GPU"
	echo "example usage: ./scripts/build.sh cpu"
	exit 1
fi

if [ "$1" == 'gpu' ]; then
	docker run --rm -itd \
		--runtime=nvidia \
		-v $PWD/$WORKDIR/Output:/workdir/model \
		-e PORT=80 \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-p 3000:80 \
		--name vfastrcnn \
		user1m/vott-reviewer-cntk:$1
else
	docker run --rm -itd \
		-v $PWD/$WORKDIR/Output:/workdir/model \
		-e PORT=80 \
		-p 3000:80 \
		--name vfastrcnn \
		user1m/vott-reviewer-cntk:$1
fi
