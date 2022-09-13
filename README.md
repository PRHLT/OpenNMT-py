# Word-Level AutoCompletion
This branch contains a toolkit for performing Word-level autocompletion. It is an extension of the [INMT branch](https://github.com/PRHLT/OpenNMT-py/tree/inmt) of the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) toolkit.

Table of Contents
=================
  * [Setup](#setup)
  * [Format](#format)
  * [Usage](#usage)
  * [Acknowledgements](#acknowledgements)
  * [Citation](#citation)

## Setup

OpenNMT-py requires:

- Python >= 3.6
- PyTorch == 1.6.0

Install `OpenNMT-py` from the sources:
```bash
git clone --branch word-level_autocompletion https://github.com/PRHLT/OpenNMT-py/
cd OpenNMT-py
pip install -e .
```
*(Optional)* Some advanced features (e.g. working pretrained models or specific transforms) require extra packages, you can install them with:

```bash
pip install -r requirements.opt.txt
```

## Format
This toolkit uses the format introduced at WMT22's [Word-Level AutoCompletion shared task](https://statmt.org/wmt22/word-autocompletion.html):

```json
{
    "src":"安全 理事会 ，",
    "context_type":"prefix",
    "left_context":"The Security",
    "right_context":"",
    "typed_seq":"Coun",
}
```

where `src` is the source sentence; `context_type` is the type of context and can have the value *prefix*, *suffix*, *bi-context* or *zero_context*; `left_context` and `right context` contain the context; and `typed_seq` is the word to autocomplete.

The expect output has the format:

```json
{
    "src":"安全 理事会 ，",
    "context_type":"prefix",
    "left_context":"The Security",
    "right_context":"",
    "typed_seq":"Coun",
    "target":"Council"
}
```

where `target` is the autocompleted word.

## Usage
Given an NMT model trained with OpenNMT-py, you can obtained the autocompletions by running the following script:

```bash
python autocomplete.py --document document.json --model nmt_model.pt \
--predictions output.pred [--bpe bpe_codes] [--wlac output.json]
```

where `document.json` contains the sentences in the format introduced above; `nmt_model.opt` is the NMT model; `output.pred` is a plain text file containing only the autompleted words (one word per input); `bpe_codes` contain the codes used for training the BPE model (if it had been used for training the NMT model); and `output.json` generates the output following the format aforementioned format.

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

The word-level autocompletion extension has been written by:
* [Ángel Navarro](https://github.com/angelnm) (PRHLT Research Center - Univesitat Politècnica de València).
* [Miguel Domingo](https://github.com/midobal) (PRHLT Research Center - Univesitat Politècnica de València).

## Citation

If you are using this toolkit for academic work, please cite the initial [system demonstration paper](https://www.aclweb.org/anthology/P17-4012) published in ACL 2017:

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

and our system submission to WMT22's [Word-Level AutoCompletion shared task](https://statmt.org/wmt22/word-autocompletion.html):

```
@inproceedings{Navarro22,
	title 		= {{PRHLT}’s Submission to {WLAC} 2022},
	author		= {Navarro, {\'A}ngel and Domingo, Miguel and Casacuberta, Francisco},
	booktitle 	= {Proceedings of the Seventh Conference on Machine Translation},
	year 		= {2022},
	note 		= {Under review.}
}
```
