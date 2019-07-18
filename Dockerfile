# parent image
FROM nvcr.io/nvidia/tensorflow:18.06-py3

MAINTAINER Sung-Lin Yeh <ff936tw@gmail.com>

# install python packages
RUN pip install --upgrade pip &&\
    pip install joblib &&\
    pip install numpy==1.16.1 &&\
    pip install pandas &&\
    pip install scikit-learn 

# install basic packages
RUN apt-get update && \
	apt-get install -y \
	tree \
	automake \
	build-essential \
	checkinstall \
	cmake \
	git \
	libtool \
	pkg-config \
	python-dev \
	sshfs \
	unzip \
	v4l-utils \
	wget \
	x264 \
	yasm

# install all the relevant libs
RUN apt-get install -y \
	libopencv-dev libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2 libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev 	libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libpng12-dev libtiff5-dev

# copy .tar.gz 
COPY opensmile-2.3.0.tar.gz .
RUN tar -xf opensmile-2.3.0.tar.gz -C /opt

# install openSMILE
RUN cd /opt/opensmile-2.3.0/ && \
	./buildWithPortAudio.sh -o /usr/local/lib && \
	./buildStandalone.sh -o /usr/local/lib

# IAAN codes
RUN git clone https://github.com/30stomercury/Interaction-aware_Attention_Network.git
