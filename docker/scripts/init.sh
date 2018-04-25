#!/bin/bash

# apt update
# apt install -y curl
# curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
# apt install -y nodejs
ln -s /workdir/api/package.json /workdir/package.json
export PATH=$PATH:/workdir/node_modules/.bin/
cp /workdir/api/models/*.py /cntk/Examples/Image/Detection/
mkdir -p /workdir/temp/inputs /workdir/temp/outputs
cd /workdir/ && npm i
cd /workdir/api/ && npm run start-dev &
bash
