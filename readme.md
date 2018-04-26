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
> docker run --rm -itd \
	-v /path/to/cntk/model/:/workdir/model/ \
	-e PORT=80 \
	-p 3000:80 \
	--name vott-reviewer \
	user1m/vott-reviewer-cntk:cpu
```

* **NOTE:** Your cntk model path must be mapped to the container's `/workdir/model/` path
	* **In the `/workdir/model/` should be a `.model` file AND a `class_map.txt` file from [your training](https://docs.microsoft.com/en-us/cognitive-toolkit/object-detection-using-faster-r-cnn#run-faster-r-cnn-on-your-own-data)**
*  This will expose an expoint on `127.0.0.1:3000/cntk`. Plug this endpoint into VOTT and review.


## Test

```
> curl \
  -F "image=@/home/user1/Desktop/test.jpg" \
  localhost:3000/cntk
```

