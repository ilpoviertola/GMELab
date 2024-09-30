# GMELab: Generative Model Evaluation Lab

An evaluation suite for your generative models. Currently supports the following metrics:

- [Frechet Audio Distance](https://github.com/gudgud96/frechet-audio-distance)
  - Supports VGGish, PANN, CLAP, AST, and EnCodec embeddings
- [ImageBind](https://github.com/facebookresearch/ImageBind) similarity score between video and audio
- [Kullback–Leibler Divergence](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/metrics/kld.py) between two audios
  - Supports [PaSST](https://arxiv.org/abs/2110.05069) embeddings
- [Sync](https://github.com/v-iashin/Synchformer) score for temporal alignment between video and audio
- Rhythm Similarity
- Spectral Contrast Similarity
- Zero Crossing Rate for audio

## Environment Setup

This code has been tested on Ubuntu 20.04 with Python 3.8.18 and PyTorch 2.2.1 using CUDA 12.1. To install the required packages, run the following command:

```bash
conda env create -f conda_env_cu12.1.yaml
conda activate gmelab_cu12.1
```

## Data

Some of the metrics are reference-based, meaning they require a reference audio or video to compare against. Make sure you have appropriate ground truth data for these metrics:

- Frechet Audio Distance*
- Kullback–Leibler Divergence
- Rhythm Similarity
- Spectral Contrast Similarity
- Zero Crossing Rate

*Not exact ground truth, but a reference audio set is required

## Usage
