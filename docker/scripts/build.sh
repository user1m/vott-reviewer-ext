#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

if [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
	echo "missing compute option CPU or GPU"
	echo "example usage: ./scripts/run.sh cpu"
	exit 1
fi

docker build -t user1m/vott-reviewer:$1 -f $SCRIPTDIR/../Dockerfile-py3-$1 $WORKDIR/docker/context
