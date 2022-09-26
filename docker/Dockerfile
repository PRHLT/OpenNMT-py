FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

##################
## Requirements ##
##################
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y git python3-pip && \
rm -rf /var/lib/apt/lists/*
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64

################
## OpenNMT-py ##
################
RUN git clone --branch lab_sessions https://github.com/PRHLT/OpenNMT-py.git /opt/opennmt-py
WORKDIR /opt/opennmt-py
RUN pip install -e . --no-cache-dir && pip install -r requirements.opt.txt && \
pip install sacrebleu

##########################
## Default: train model ##
##########################
CMD onmt_train -config config.yaml
