FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# This fix: libGL error: No matching fbConfigs or visuals found
ENV LIBGL_ALWAYS_INDIRECT=1

# Install Python 3
RUN apt-get update && apt-get install -y libopencv-dev python3-opencv \
    && apt-get install -y python3-numpy

COPY app/ .

#start script inside the container
CMD ["python3", "./QFISHbio-preprocess.py"]


# build container
# docker build -t python-pyqt .

# run container
# docker run --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=172.27.176.1:0.0 ydongmof/app_preprocess:final