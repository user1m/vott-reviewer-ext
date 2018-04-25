#!/bin/bash

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
WORKDIR=$SCRIPTDIR/../..

docker run --rm -it \
	-v $PWD/$WORKDIR/Output:/workdir/output \
	-v $PWD/$WORKDIR/api:/workdir/api \
	-v $PWD/$WORKDIR/flaskapi:/workdir/flaskapi \
	-v $PWD/$SCRIPTDIR/init.sh:/scripts/init.sh \
	-e PORT=80 \
	-p 3000:80 \
	--name vfastrcnn \
	vott-fastrcnn \
	bash -c "/scripts/init.sh"
