#FROM ubuntu:20.04
FROM spmallick/opencv-docker:opencv
FROM nvidia/cuda:11.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root

RUN apt update && apt install -y sudo cmake g++ wget unzip python3-pip python3-dev \
    python-dev python3-venv \
    ffmpeg libsm6 libxext6 git

# COPY . $HOME/pvdn
# WORKDIR $HOME/pvdn

RUN pip3 install --upgrade pip && \
    pip3 install git+https://github.com/larsOhne/pvdn.git
    # cd pvdn/detection/utils && \
    # g++ -fpic -shared -o image_operations.so image_operations.cpp HeadLampObject.cpp
