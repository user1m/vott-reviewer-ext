#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

if [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
	echo "missing compute option CPU or GPU"
	echo "example usage: ./scripts/build.sh cpu"
	exit 1
fi

if [ "$1" == 'gpu' ]; then
	nvidia-docker build -t user1m/vott-reviewer-cntk:$1 -f $SCRIPTDIR/../Dockerfile-py3-$1 $WORKDIR/docker
else
	docker build -t user1m/vott-reviewer-cntk:$1 -f $SCRIPTDIR/../Dockerfile-py3-$1 $WORKDIR/docker
fi
