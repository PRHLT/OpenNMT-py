# Run through Docker

## Table of contents

* [Image building](#image-building).
* [Structure](#structure).
* [Dataset](#dataset).
* [Config](#config).
* [Vocabulary](#vocabulary).
* [Training](#trainig).
* [Translation](#translation).
* [Evaluation](#evaluation).
* [Tunning](#Tuning).

## Image building
The first step is to build the image. You can do so by running the following command (assuming that the current directory is the one in which this repo has been cloned. Otherwise `.` should be replaced with the correct path):

```
docker build -t opennmt-py-lab .
```

## Structure
For simplicity, we are going to assume that we have a folder `data` in which to store both the dataset and the models and that it is located in `$(pwd)`. This folder contains the following structure:

* dataset/
  * tr.src.
  * tr.tgt.
  * dev.src.
  * dev.tgt.
  * test.src.
  * test.tgt.
* models/

where `dataset` is the folder in which the dataset is stored; `src` is the source language and `tgt` is the target language. **For the lab session, this will be covered in the next section.**

## Dataset
The dataset is located at `dataset/EuTrans`. It is already set up and no further preprocesses are needed. However, we will need to create a copy in the `data` folder:

```
mkdir ~/TA/Practica2/data
cd ~/TA/Practica2
cp -r ${repo}/dataset/EuTrans data/
```

where `${repo}` is the path to this repo.

## Config
The default config file is located at `${repo}/docker/config.yaml`. You should copy this file to your running directory:

```
cp ${repo}/docker/config.yaml .
```
where `${repo}` is the path to this repo.

## Vocabulary
Prior to training, you need to build the vocabulary:

```
docker container run -it --rm -v "$(pwd)"/data:/data \
-v "$(pwd)"/config.yaml:/opt/opennmt-py/config.yaml opennmt-py-lab \
onmt_build_vocab -config config.yaml
```

where `-n_sample` represents the number of lines sampled from each corpus to build the vocabulary.

## Train
You can train a model by running the following command:

```
docker container run -it --rm -v "$(pwd)"/data:/data \
-v "$(pwd)"/config.yaml:/opt/opennmt-py/config.yaml opennmt-py-lab
```

If you want to train using a GPU, you should uncomment the denoted lines from the config file and add the flag `--gpus all` to the run command:

```
docker container run -it --rm --gpus all -v "$(pwd)"/data:/data \
-v "$(pwd)"/config.yaml:/opt/opennmt-py/config.yaml opennmt-py-lab
```

## Translate
To translate a document, you just need to run the following command:

```
docker container run -it --rm -v "$(pwd)"/data:/data \
-v "$(pwd)"/config.yaml:/opt/opennmt-py/config.yaml opennmt-py-lab onmt_translate \
-model /data/models/model_step_$n.pt -src /data/dataset/test.src -output /data/test.hyp \
-verbose -replace_unk
```

where `model_step_$n.pt` is the desired model to use and `/data/dataset/test.src` is the document to translate (`EuTrans/test.es` in the case of the lab session).

Alternatively, if you want to use a GPU, you can translate a document by doing:

```
docker container run -it --rm --gpus all -v "$(pwd)"/data:/data \
-v "$(pwd)"/config.yaml:/opt/opennmt-py/config.yaml opennmt-py-lab onmt_translate \
-model /data/models/model_step_$n.pt -src /data/dataset/test.src -output /data/test.hyp \
-gpu 0 -verbose -replace_unk
```

## Evaluation
You can evaluate a translation hypothesis by doing:

```
docker container run -i --rm -v "$(pwd)"/data:/data \
-v "$(pwd)"/config.yaml:/opt/opennmt-py/config.yaml opennmt-py-lab \
sacrebleu --force -f text /data/dataset/test.tgt < data/test.hyp
```

where `/data/dataset/test.tgt` is the reference translation (`/data/EuTrans/test.en` in the case of the lab session).
