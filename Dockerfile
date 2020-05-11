ARG TF_VER="2.2.0"
FROM tensorflow/tensorflow:${TF_VER}

RUN apt update && apt install -y libopenjp2-7-dev libdcmtk-dev curl cmake checkinstall

ARG BUILD_VER="TBD"
ENV BUILD_VER=${BUILD_VER}

RUN cd /tmp && \
    curl -L  https://github.com/DraconPern/fmjpeg2koj/archive/master.tar.gz | tar xvz && \
    cd fmjpeg2koj-master && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr . && \
    make -j4 && \
    checkinstall -D -y \
      --install=no \
      --fstrans=yes \
      --reset-uids=yes \
      --pkgname=fmjpeg2koj \
      --pkgversion=${BUILD_VER} \
      --pkgrelease="tar.bz2" \
      --arch=amd64 \
      --pkglicense=Apache \
      --maintainer="TBD" \
      --requires=libopenjp2-7-dev,libdcmtk-dev && \
    dpkg -i fmjpeg2koj_${BUILD_VER}-tar.bz2_amd64.deb && \
    mkdir dist && \
    mv *.deb dist

RUN mkdir /tmp/build
WORKDIR /tmp/build

COPY  gradient_decode_dicom gradient_decode_dicom
COPY  setup.cfg setup.cfg
COPY  setup.py setup.py
COPY  README.md README.md
COPY  test.cc test.cc


RUN ls -lah /tmp/build/

RUN python3 setup.py bdist_wheel
RUN cd dist && python3 -m pip install *.whl && cd ..

COPY test test
ARG CACHEBUST=1

RUN cd test && python test.py && cd ..
