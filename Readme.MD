It's the code for "Efficient Model of Temporal Sentence Video Grounding with Multi-Teacher Knowledge Distillation"


## Preparation
The visual features of `Charades-STA`, `ActivityNet Captions`, and `TACoS` are available at [Box Drive](
https://app.box.com/s/d7q5atlidb31cuj1u8znd7prgrck1r1s).
Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip). Modify the path setting in `./configs/charades/SeqPAN.yaml`.

## Prerequisites
- python3 with tensorflow>=`2.0`, pytorch, tqdm, nltk, numpy, eta.

## Quick Start
**Train**
```shell script

# train
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/MultiTeacherEnsemble.yaml
# test
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/MultiTeacherEnsemble.yaml --mode test
# summary
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/MultiTeacherEnsemble.yaml --mode summary

```