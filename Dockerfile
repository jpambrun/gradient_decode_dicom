ARG TF_VER="1.13.1"

FROM tensorflow/tensorflow:${TF_VER}-py3
ARG TF_VER="1.13.1"
RUN apt update && apt install -y libdcmtk-dev
ENV VER=${TF_VER}
RUN mkdir /tmp/build
WORKDIR /tmp/build
COPY . /tmp/build
RUN python3 setup.py bdist_wheel
RUN cd dist && python3 -m pip install *.whl && cd ..
RUN cd test && python test.py && cd ..
