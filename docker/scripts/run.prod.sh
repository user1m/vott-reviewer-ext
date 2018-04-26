#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

if [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
	echo "missing compute option CPU or GPU"
	echo "example usage: ./scripts/build.sh cpu"
	exit 1
fi

docker run --rm -itd \
	-v $PWD/$WORKDIR/Output:/workdir/model \
	-e PORT=80 \
	-p 3000:80 \
	--name vfastrcnn \
	user1m/vott-reviewer-cntk:$1
# bash
