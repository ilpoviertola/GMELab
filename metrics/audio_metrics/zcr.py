from pathlib import Path
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from eval_utils.dataset import AudioWithGtDataset
from torch.utils.data import DataLoader


def calculate_zcr(
    samples_dir: Path,
    gt_dir: Path,
    verbose: bool = False,
    metadata: Optional[Dict[str, float]] = None,
    duration: float = 2.56,
    apply_metadata_to_samples: bool = False,
) -> float:
    dataset = AudioWithGtDataset(
        samples_dir, gt_dir, duration, metadata, apply_metadata_to_samples
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    total_zcr_similarity = 0.0
    count = 0

    for batch in tqdm(loader, desc="Calculating ZCR"):
        sample = batch["sample_audio"].numpy()
        gt = batch["gt_audio"].numpy()
        gt_zcr = np.mean(np.abs(np.diff(np.sign(gt))) > 0)
        sample_zcr = np.mean(np.abs(np.diff(np.sign(sample))) > 0)
        zcr_similarity = 1 - np.abs(gt_zcr - sample_zcr)
        total_zcr_similarity += zcr_similarity
        count += 1

    if verbose:
        print(f"ZCR similarity: {total_zcr_similarity / count}")

    return float(total_zcr_similarity / count)
