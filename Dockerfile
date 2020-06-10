FROM nvcr.io/nvidia/tensorrt:19.12-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender1 \
    qt5-default \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip
RUN python3 -m pip install --no-cache-dir -U setuptools

WORKDIR /tmp
COPY ./client_engine_requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r requirements.txt

ENV CMAKE_VERSION="3.16.4"
RUN wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
	tar -xzf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
	export PATH=$PWD/cmake-${CMAKE_VERSION}-Linux-x86_64/bin:$PATH

RUN git clone --depth 10 --branch 4.0.1 https://github.com/opencv/opencv opencv && \
    git clone --depth 10 --branch 4.0.1 https://github.com/opencv/opencv_contrib.git opencv_contrib && \
    mkdir -p opencv/build && cd opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_TBB=ON \
          -D WITH_V4L=ON \
          -D WITH_QT=ON \                   
          -D WITH_OPENCL=ON \
          -D WITH_GTK=ON \
          -D WITH_LIBV4L=ON \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D WITH_FFMPEG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          .. && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2

COPY . .

RUN cd clothes_seg/src/cytho_lib && \
    python3 setup.py build_ext --inplace && \
    cp *.so ../../..


# docker build -t cloth_seg_trt .
# docker run --rm  -it --runtime=nvidia -v ~/Desktop/temp:/py -w /py cloth_seg_trt bash
