from pathlib import Path
from typing import Dict, Optional

import numpy as np
import librosa
from tqdm import tqdm

from eval_utils.dataset import AudioWithGtDataset
from torch.utils.data import DataLoader


def calculate_rhythm_similarity(
    samples_dir: Path,
    gt_dir: Path,
    sample_rate: int,
    verbose: bool = False,
    metadata: Optional[Dict[str, float]] = None,
    duration: float = 2.56,
    apply_metadata_to_samples: bool = False,
) -> float:
    dataset = AudioWithGtDataset(
        samples_dir, gt_dir, duration, metadata, apply_metadata_to_samples
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    total_rhythm_similarity = 0.0
    count = 0

    for batch in tqdm(loader, desc="Calculating rhythm similarity"):
        sample = batch["sample_audio"][0, 0, :].numpy()
        gt = batch["gt_audio"][0, 0, :].numpy()

        min_length = len(gt)
        gt_onset_vector = np.zeros(min_length)
        gt_onsets = librosa.onset.onset_detect(y=gt, sr=sample_rate, units="time")
        gt_onsets = np.array(gt_onsets) * sample_rate
        gt_onsets = gt_onsets[gt_onsets < min_length]
        gt_onset_vector[gt_onsets.astype(int)] = 1

        min_length = min(min_length, len(sample))
        sample_onset_vector = np.zeros(min_length)
        sample_onsets = librosa.onset.onset_detect(
            y=sample[:min_length], sr=sample_rate, units="time"
        )
        sample_onsets = np.array(sample_onsets) * sample_rate
        sample_onsets = sample_onsets[sample_onsets < min_length]
        sample_onset_vector[sample_onsets.astype(int)] = 1

        rhythm_similarity = (
            np.corrcoef(
                gt_onset_vector[:min_length],
                sample_onset_vector[:min_length],
            )[0, 1]
            + 1
        ) / 2
        total_rhythm_similarity += (
            rhythm_similarity if not np.isnan(rhythm_similarity) else 0
        )
        count += 1

    if verbose:
        print(f"Rhythm similarity: {total_rhythm_similarity / count}")

    return float(total_rhythm_similarity / count)
