.DEFAULT_GOAL := all
.PHONY: build clean

clean:
	rm -fr dist || true

build:
	docker build .  --build-arg CACHEBUST=$(shell date +%s)  -t tf-dicom:tf2 --build-arg TF_VER="2.1.0-py3"
	id=$$(docker create tf-dicom:tf2) && docker cp $$id:/tmp/build/dist . && docker rm -v $$id
	id=$$(docker create tf-dicom:tf2) && docker cp $$id:/tmp/fmjpeg2koj-master/dist . && docker rm -v $$id
