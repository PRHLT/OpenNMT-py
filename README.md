# Interactive Neural Machine Translation

This branch extends [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) to provide a back-end for performing several Interactive Machine Translation (IMT) protocols.

## Table of Contents
  * [Setup](#setup)
  * [IMT Protocols](#imt-protocols)
  * [Model Training](#model-training)
  * [Acknowledgements](#acknowledgements)
  * [Citation](#citation)

## Setup

OpenNMT-py requires:

- Python >= 3.6
- PyTorch == 1.6.0

Install `OpenNMT-py` from the sources:
```bash
git clone --inmt https://github.com/PRHLT/OpenNMT-py/
cd OpenNMT-py
pip install -e .
```

Note: if you encounter a `MemoryError` during installation, try to use `pip` with `--no-cache-dir`.

*(Optional)* Some advanced features (e.g. working pretrained models or specific transforms) require extra packages, you can install them with:

```bash
pip install -r requirements.opt.txt
```

## IMT Protocols

### Prefix-based IMT
This protocol implements the prefix-based framework introduced by [Barrachina et al. (2009)](https://aclanthology.org/J09-1002/).

### Segment-based IMT
This protocol implements the segment-based framework introduced by [Domingo et al. (2017)](https://link.springer.com/article/10.1007/s10590-017-9213-3?error=cookies_not_supported&code=7af7856a-375e-405b-ab83-5de11564413d) and [Peris et al. (2017)](https://www.sciencedirect.com/science/article/pii/S0885230816301000).

## Simulation

Since conducting frequent human evaluations at the development stage has a high time and economic costs, we offered a simulated environment in which the user aims to generate the translations from a reference file. You can run a simulation as follows:

```
onmt_simulation --model ${model} --src ${file} --tgt ${ref_file} {options}
```

where:

* `${model}` is the desired NMT model to use.
* `${file}` is the file to translate.
* `${ref_file}` is the file containing the reference translations.

The following options are available:

```
--inmt_verbose, -inmt_verbose
                        Increase the verbose level of the simulation.
--character_level, -character_level
                        Perform the simulation generating new hypothesis 
                        each time the user types a character. 
                        (Default: word-level.)
--segment, -segment   
                        Use the segment-based protocol. 
                        (Default: prefix-based.)
```

## Model Training

Currently, this project is based on *OpenNMT-py 2.0*. Thus, we can only ensure compatibility with models trained with this version of *OpenNMT-py*. Therefore, to avoid compatibility issues, it is recommended to train the models using this branch.

### Prepare the data

To get started, we propose to download a toy English-German dataset for machine translation containing 10k tokenized sentences:

```bash
wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
tar xf toy-ende.tar.gz
cd toy-ende
```

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are used to evaluate the convergence of the training. It usually contains no more than 5k sentences.

```text
$ head -n 2 toy-ende/src-train.txt
It is not acceptable that , with the help of the national bureaucracies , Parliament &apos;s legislative prerogative should be made null and void by means of implementing provisions whose content , purpose and extent are not laid down in advance .
Federal Master Trainer and Senior Instructor of the Italian Federation of Aerobic Fitness , Group Fitness , Postural Gym , Stretching and Pilates; from 2004 , he has been collaborating with Antiche Terme as personal Trainer and Instructor of Stretching , Pilates and Postural Gym .
```

We need to build a **YAML configuration file** to specify the data that will be used:

```yaml
# toy_en_de.yaml

## Where the samples will be written
save_data: toy-ende/run/example
## Where the vocab(s) will be written
src_vocab: toy-ende/run/example.vocab.src
tgt_vocab: toy-ende/run/example.vocab.tgt
# Prevent overwriting existing files in the folder
overwrite: False

# Corpus opts:
data:
    corpus_1:
        path_src: toy-ende/src-train.txt
        path_tgt: toy-ende/tgt-train.txt
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
...

```

From this configuration, we can build the vocab(s) that will be necessary to train the model:

```bash
onmt_build_vocab -config toy_en_de.yaml -n_sample 10000
```

**Notes**:
- `-n_sample` is required here -- it represents the number of lines sampled from each corpus to build the vocab.
- This configuration is the simplest possible, without any tokenization or other *transforms*. See [other example configurations](https://github.com/OpenNMT/OpenNMT-py/tree/master/config) for more complex pipelines.

### Train the model

To train a model, we need to **add the following to the YAML configuration file**:
- the vocabulary path(s) that will be used: can be that generated by onmt_build_vocab;
- training specific parameters.

```yaml
# toy_en_de.yaml

...

# Vocabulary files that were just created
src_vocab: toy-ende/run/example.vocab.src
tgt_vocab: toy-ende/run/example.vocab.tgt

# Train on a single GPU
world_size: 1
gpu_ranks: [0]

# Where to save the checkpoints
save_model: toy-ende/run/model
save_checkpoint_steps: 500
train_steps: 1000
valid_steps: 500

```

Then you can simply run:

```bash
onmt_train -config toy_en_de.yaml
```

This configuration will run the default model, which consists of a 2-layer LSTM with 500 hidden units on both the encoder and decoder. It will run on a single GPU (`world_size 1` & `gpu_ranks [0]`).

Before the training process actually starts, the `*.vocab.pt` together with `*.transforms.pt` will be dumpped to `-save_data` with configurations specified in `-config` yaml file. We'll also generate transformed samples to simplify any potentially required visual inspection. The number of sample lines to dump per corpus is set with the `-n_sample` flag.

For more advanded models and parameters, see [other example configurations](https://github.com/OpenNMT/OpenNMT-py/tree/master/config) or the [FAQ](https://opennmt.net/OpenNMT-py/FAQ.html).

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC) to reproduce OpenNMT-Lua using PyTorch.

Major contributors are:
* [Sasha Rush](https://github.com/srush) (Cambridge, MA)
* [Vincent Nguyen](https://github.com/vince62s) (Ubiqus)
* [Ben Peters](http://github.com/bpopeters) (Lisbon)
* [Sebastian Gehrmann](https://github.com/sebastianGehrmann) (Harvard NLP)
* [Yuntian Deng](https://github.com/da03) (Harvard NLP)
* [Guillaume Klein](https://github.com/guillaumekln) (Systran)
* [Paul Tardy](https://github.com/pltrdy) (Ubiqus / Lium)
* [François Hernandez](https://github.com/francoishernandez) (Ubiqus)
* [Linxiao Zeng](https://github.com/Zenglinxiao) (Ubiqus)
* [Jianyu Zhan](http://github.com/jianyuzhan) (Shanghai)
* [Dylan Flaute](http://github.com/flauted) (University of Dayton)
* ... and more!

OpenNMT-py is part of the [OpenNMT](https://opennmt.net/) project.

The IMT extension has been written by:
* [Ángel Navarro](https://github.com/angelnm) (PRHLT Research Center - Univesitat Politècnica de València).
* [Miguel Domingo](https://github.com/midobal) (PRHLT Research Center - Univesitat Politècnica de València).

## Citation

If you are using OpenNMT-py for academic work, please cite the initial [system demonstration paper](https://www.aclweb.org/anthology/P17-4012) published in ACL 2017:

```
@inproceedings{klein-etal-2017-opennmt,
    title = "{O}pen{NMT}: Open-Source Toolkit for Neural Machine Translation",
    author = "Klein, Guillaume  and
      Kim, Yoon  and
      Deng, Yuntian  and
      Senellart, Jean  and
      Rush, Alexander",
    booktitle = "Proceedings of {ACL} 2017, System Demonstrations",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-4012",
    pages = "67--72",
}
```
