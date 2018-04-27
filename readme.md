# VOTT Reviewer Ext


## Prereqs

* [VOTT](https://github.com/Microsoft/VoTT)
* [docker](https://docs.docker.com/install/)

This is a tool to run a model reviewer in a docker container, to decouple model reviewing from the VOTT application, in order to keep the application light-weight


## Setup

```
> docker pull user1m/vott-reviewer-cntk:cpu
```


## Usage

```
> git clone https://github.com/User1m/vott-reviewer-ext.git
> cd vott-reviewer-ext
> export MODEL_PATH=/path/to/cntk/model/
> ./docker/scripts/run.prod.sh cpu (or gpu)
```

* **NOTE:** Your cntk model path must be mapped to the container's `/workdir/model/` path
* **In the `/workdir/model/` should be a `.model` file AND a `class_map.txt` file from [your training](https://docs.microsoft.com/en-us/cognitive-toolkit/object-detection-using-faster-r-cnn#run-faster-r-cnn-on-your-own-data)**
*  This will expose an expoint on `127.0.0.1:3000/cntk`. Plug this endpoint into VOTT and review.


## Know GPU Machine Issues

**NOTE** - If you have a GPU machine w/ nvidia installed you might run into an issue w/ `nvidia-docker` & creating the container. This is due to the fact that the host already contains `nvidia-cuda-toolkit` binaries, found in `ls -la /usr/bin/nvidia-*`, and the container tries to mount it's `nvidia` binaries over the existing ones.

A current workaround is to rename some of the host `nvidia` files found in `/usr/bin/nvidia-*` so that the container can be created, then you can rename them back after container is created.

#### !! See [run.prod.sh](https://github.com/User1m/vott-reviewer-ext/blob/master/docker/scripts/run.prod.sh) for an example of this.


## Test

```
> curl \
  -F "image=@/home/user1/Desktop/test.jpg" \
  localhost:3000/cntk
```

