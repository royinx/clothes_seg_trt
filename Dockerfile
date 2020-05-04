FROM nvcr.io/nvidia/tensorrt:19.12-py3

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  libsm6 \
  libxext6 \
  libxrender1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip
RUN python3 -m pip install --no-cache-dir -U setuptools

WORKDIR /tmp
COPY ./client_engine_requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r requirements.txt

# docker build -t cloth_seg_trt .
# docker run --rm  -it --runtime=nvidia -v ~/Desktop/temp:/py -w /py cloth_seg_trt bash
