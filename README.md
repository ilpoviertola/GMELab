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

## Submodules

This repository uses submodules. To clone the repository with submodules, run the following command:

```bash
git clone --recurse-submodules
```

To update the submodules, run the following command:

```bash
git submodule update --init --recursive
```

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

### 1. Generate Data

Use your generative model of choice to generate audio and video samples.

### 2. Configure GMELab pipeline

Use `.configs/generate_evaluation_configs.py` to generate the pipeline configuration file. This file will define the following:

- gt_directory: Path to the directory containing the ground truth data. Used with metrics specified above in [Data](#data) section.
- id: Unique identifier for the experiment.
- metadata: Metadata for the experiment. See .data/metadata/*.csv for examples.
- pipeline: The actual evaluation pipeline. Define the metrics you want to use and their configurations. See `configs/evaluation_cfg.py` for metric-specific configurations.
- sample_directory: Path to the directory containing the generated samples.
- verbose: Whether to print verbose output.

### 3. Run GMELab

Use the following command to run the evaluation pipeline:

```bash
python run_evaluation.py -p <path_to_pipeline_config> -td <path_to_table_output_directory> -pd <path_to_plot_output_directory>
```

You can also provide path to a directory containing multiple pipeline configurations. E.g.

```bash
python run_evaluation.py -p ./configs/V-AURA/VisualSound -td ./configs/V-AURA/VisualSound/ -pd ./configs/V-AURA/VisualSound/
```

Each evaluation run will generate a results YAML-file in the sample directory. The results will be saved in the table output directory as a CSV file and the plots will be saved in the plot output directory, if specified.
