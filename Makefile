TF_VER=2.15.0.post1
BUILD_VER=2.15.0-1

.DEFAULT_GOAL := all
.PHONY: build clean

clean:
	rm -fr dist || true

build:
	docker build .  --progress=plain --build-arg CACHEBUST=$(shell date +%s)  -t tf-dicom:tf2 --build-arg TF_VER=$(TF_VER) --build-arg BUILD_VER=$(BUILD_VER)
	id=$$(docker create tf-dicom:tf2) && docker cp $$id:/tmp/build/dist . && docker rm -v $$id
	id=$$(docker create tf-dicom:tf2) && docker cp $$id:/tmp/fmjpeg2koj-master/dist . && docker rm -v $$id
