#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

docker run --rm -it \
	-v $PWD/$WORKDIR/Output:/workdir/model \
	-v $PWD/$WORKDIR/docker/context/flaskapi:/workdir/flaskapi \
	-v $PWD/$WORKDIR/pyfiles:/workdir/pyfiles \
	-v $PWD/$WORKDIR/scripts/:/scripts/ \
	-e PORT=80 \
	-p 3000:80 \
	--name vfastrcnn \
	user1m/vott-reviewer-cntk:cpu \
	bash -c "/scripts/init.sh"

# -v $PWD/$WORKDIR/api:/workdir/api \
