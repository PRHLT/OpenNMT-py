#!/bin/bash

if [[ "$#" -ne 1 ]]
then
    >&2 echo "Usage: $0 installation_path."
    exit 1

elif [[ ! -d "$1" ]]
then
    >&2 echo "Error: path ${1} does not exist."
    exit 1
fi

export PTH=${1}/NMT_TA
mkdir -p "$PTH"
if [[ ! -d "$PTH" ]]
then
    >&2 echo "Error: installation directory could not be created. You may need sudo permisions to install in $1."
    exit 1
fi

##########################
# OpenNMT-py installation #
##########################
git clone --branch lab_sessions https://github.com/PRHLT/OpenNMT-py.git ${PTH}/OpenNMT-py

##########################
# miniconda installation #
##########################
wget https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p ${PTH}/miniconda
export conda=${PTH}/miniconda/bin/conda
hash -r
${conda} config --set always_yes yes --set changeps1 no
${conda} update -q conda
${conda} info -a
${conda} install pip
export pip=${PTH}/miniconda/bin/pip
${pip} install sacrebleu
${pip} install -e ${PTH}/OpenNMT-py
${pip} install -r ${PTH}/OpenNMT-py/requirements.opt.txt
rm miniconda.sh
