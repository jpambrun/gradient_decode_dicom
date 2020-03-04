.DEFAULT_GOAL := all
.PHONY: build clean

clean:
	rm -fr dist || true

build:
	docker build . -t tf-dicom:tf2 --build-arg TF_VER="2.0.1"
	id=$$(docker create tf-dicom:tf2) && docker cp $$id:/tmp/build/dist . && docker rm -v $$id
	id=$$(docker create tf-dicom:tf2) && docker cp $$id:/tmp/fmjpeg2koj-master/dist . && docker rm -v $$id
