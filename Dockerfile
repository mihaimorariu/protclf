FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

RUN apt-get update --fix-missing -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake build-essential python3-pip python3

COPY . /home/protclf
RUN cd /home/protclf && pip3 install -r requirements.txt
RUN echo "export PYTHONPATH=/home/protclf" > ~/.bashrc
