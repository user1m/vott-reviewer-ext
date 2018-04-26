#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

if [ "$1" != "cpu" ] && [ "$1" != "gpu" ]; then
	echo "missing compute option CPU or GPU"
	echo "example usage: ./scripts/build.sh cpu"
	exit 1
fi

if [ "$1" == 'gpu' ]; then
	docker run --rm -it \
		--runtime=nvidia \
		-v $PWD/$WORKDIR/Output:/workdir/model \
		-v $PWD/$WORKDIR/docker/context/flaskapi:/workdir/flaskapi \
		-v $PWD/$WORKDIR/pyfiles:/workdir/pyfiles \
		-v $PWD/$WORKDIR/scripts/:/scripts/ \
		-e PORT=80 \
		-p 3000:80 \
		--name vfastrcnn \
		user1m/vott-reviewer-cntk:$1 \
		bash -c "/scripts/init.sh"
else
	docker run --rm -it \
		-v $PWD/$WORKDIR/Output:/workdir/model \
		-v $PWD/$WORKDIR/docker/context/flaskapi:/workdir/flaskapi \
		-v $PWD/$WORKDIR/pyfiles:/workdir/pyfiles \
		-v $PWD/$WORKDIR/scripts/:/scripts/ \
		-e PORT=80 \
		-p 3000:80 \
		--name vfastrcnn \
		user1m/vott-reviewer-cntk:$1 \
		bash -c "/scripts/init.sh"
fi
# -v $PWD/$WORKDIR/api:/workdir/api \
