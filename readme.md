# VOTT Reviewer Ext


## Prereqs

* [VOTT](https://github.com/Microsoft/VoTT)
* [docker](https://docs.docker.com/install/)

This is a tool to run a model reviewer in a docker container to decouple model reviewing from the VOTT application to keep the application light-weight


## Setup

```
> docker pull user1m/vott-reviewer-cntk-cpu
```


## Usage

```
> docker run --rm -d \
	-v /path/to/cntk/model/:/workdir/model \
	-e PORT=80 \
	-p 3000:80 \
	--name vottreviewer \
	user1m/vott-reviewer-cntk 
```

*  This will expose an expoint on `127.0.0.1:3000/cntk`. Plug this endpoint into VOTT