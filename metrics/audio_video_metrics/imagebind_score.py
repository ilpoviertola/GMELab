from pathlib import Path
from math import floor

import torch
import numpy as np
from tqdm import tqdm

from submodules.ImageBind.imagebind.data import (
    load_and_transform_video_data,
    load_and_transform_audio_data,
)
from submodules.ImageBind.imagebind.models import imagebind_model
from submodules.ImageBind.imagebind.models.imagebind_model import ModalityType


BATCH_SIZE = 3


def calculate_imagebind_score(
    video_dir: Path,
    device: str,
    get_diagonal_scores: bool = True,
    afps: int = 16000,
    verbose: bool = False,
):
    # get videos
    all_videos = list(video_dir.glob("*.mp4"))

    # load model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    running_score = 0
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    # run model inference
    for i in tqdm(
        range(0, len(all_videos), BATCH_SIZE), desc="Calculating ImageBind score"
    ):
        # load video and audio data
        try:
            video_data = load_and_transform_video_data(
                all_videos[i : i + BATCH_SIZE],
                device,
                sample_rate=afps,
            )
            audio_data = load_and_transform_audio_data(
                all_videos[i : i + BATCH_SIZE],
                device,
                sample_rate=afps,
            )
        except Exception as e:
            print(e)
            continue
        inputs = {
            ModalityType.AUDIO: audio_data,
            ModalityType.VISION: video_data,
        }

        with torch.no_grad():
            embeddings = model(inputs)

        sim_scores = cos_sim(
            embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]
        )
        sim_scores = sim_scores.cpu().numpy()
        running_score += np.sum(sim_scores)

    score = running_score / len(all_videos)
    if verbose:
        print("ImageBind score:", score)
    return float(score)
