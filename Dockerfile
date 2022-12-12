FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install non PiPy dependecies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    g++ python3\
    python3-pip python-is-python3 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/ 
RUN pip install --no-cache-dir --upgrade pip

# Create user inside Container
ARG USER=pvdn
ARG UID=1000
ARG GID=1000

RUN useradd -m ${USER} --uid=${UID}
USER ${UID}:${GID}

# Copy Source
COPY --chown=${UID}:${GID} . /source
WORKDIR /source

# Install package
RUN pip install --no-cache-dir --no-warn-script-location -e .
RUN python -m unittest discover -s tests -p "*.py"
