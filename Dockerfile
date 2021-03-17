FROM nvidia/cuda:11.0-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y sudo cmake g++ wget unzip python3-pip python3-dev python-dev \
    ffmpeg libsm6 libxext6


RUN mkdir -p ~/opencv cd ~/opencv && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir -p build && cd build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master && \
    cmake --build .

COPY . /usr/src
WORKDIR /usr/src
RUN pip3 install -e . && \
    cd pvdn/detection/utils && \
    g++ -fpic -shared -o image_operations.so image_operations.cpp HeadLampObject.cpp
