#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

docker run --rm -itd \
	-v $PWD/$WORKDIR/Output:/workdir/model \
	-e PORT=80 \
	-p 3000:80 \
	--name vfastrcnn \
	user1m/vott-reviewer:cpu \
	# bash

