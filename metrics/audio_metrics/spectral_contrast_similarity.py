from pathlib import Path
from typing import Dict, Optional

import numpy as np
import librosa
from tqdm import tqdm

from eval_utils.dataset import AudioWithGtDataset
from torch.utils.data import DataLoader


def calculate_spectral_contrast_similarity(
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
    total_spectral_contrast_similarity = 0.0
    count = 0

    for batch in tqdm(loader, desc="Calculating spectral contrast similarity"):
        sample = batch["sample_audio"].numpy()
        gt = batch["gt_audio"].numpy()
        gt_spectral_contrast = librosa.feature.spectral_contrast(y=gt, sr=sample_rate)
        sample_spectral_contrast = librosa.feature.spectral_contrast(
            y=sample, sr=sample_rate
        )
        min_columns = min(
            gt_spectral_contrast.shape[1], sample_spectral_contrast.shape[1]
        )
        sample_spectral_contrast = sample_spectral_contrast[:, :min_columns]
        gt_spectral_contrast = gt_spectral_contrast[:, :min_columns]
        spectral_contrast_similarity = np.mean(
            np.abs(sample_spectral_contrast - gt_spectral_contrast)
        )
        normalized_spectral_contrast_similarity = spectral_contrast_similarity / np.max(
            [np.abs(gt_spectral_contrast), np.abs(sample_spectral_contrast)]
        )
        total_spectral_contrast_similarity += normalized_spectral_contrast_similarity
        count += 1

    if verbose:
        print(
            f"Spectral contrast similarity: {total_spectral_contrast_similarity / count}"
        )

    return float(total_spectral_contrast_similarity / count)
