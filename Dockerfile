FROM ubuntu:16.04

RUN apt-get update
RUN apt install -y sudo graphviz python-pip curl git
RUN pip install --upgrade pip jupyter graphviz meshcat numpy
RUN git clone https://github.com/RussTedrake/underactuated /underactuated
RUN yes | sudo /underactuated/scripts/setup/ubuntu/16.04/install_prereqs
RUN apt install -y python-tk xvfb mesa-utils libegl1-mesa libgl1-mesa-glx libglu1-mesa libx11-6 x11-common x11-xserver-utils
RUN curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-20180817-xenial.tar.gz && sudo tar -xzf drake.tar.gz -C /opt
RUN yes | sudo /opt/drake/share/drake/setup/install_prereqs
ENV PYTHONPATH=/underactuated/src:/opt/drake/lib/python2.7/site-packages
COPY ./ /test_dir

ENTRYPOINT bash -c "/test_dir/start_docker.sh && /bin/bash"
